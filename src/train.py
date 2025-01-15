
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import math
from datetime import datetime
import torch
import random
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.


def create_envs(num_envs=2, max_steps=200):
    envoriments= [TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=max_steps) for _ in range(num_envs-1)]
    # print("size envoriments", len(envoriments))
    env_non_random =TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=max_steps) 
    envoriments.append(env_non_random)
    # print("size envoriments", len(envoriments))
    return envoriments

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


class ProjectAgent:
    def __init__(self):
        self.episode_rewards = []
        self.Actions = []
        self.States = []
        self.NextStates = []
        self.Rewards = []
        self.Dones = []
        self.Qfunction = None
        self.gamma = 0.98
        self.nb_actions = 4  
        self.nb_states = 6  
        self.residuals = []
        self.buffer_size = 40000

    def act(self, observation, use_random=False):
        if use_random or self.Qfunction is None:
            return env.action_space.sample()
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(observation, a).reshape(1, -1)
            Qsa.append(self.Qfunction.predict(sa)[0])
        return np.argmax(Qsa)
    
    def act_train(self,env, observation, use_random=False):
        if use_random or self.Qfunction is None:
            return env.action_space.sample()
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(observation, a).reshape(1, -1)
            Qsa.append(self.Qfunction.predict(sa)[0])
        return np.argmax(Qsa)

    def save(self, path):
        # compressed model, since file becomes too long to push
        joblib.dump(self.Qfunction, path, compress=3)

    def load(self, path="compressed_model.pkl"):
        self.Qfunction = joblib.load(path)

    def append_samples(self, state, action, reward, next_state, done):
        if len(self.States) >= self.buffer_size:
            self.States.pop(0)
            self.Rewards.pop(0)
            self.Actions.pop(0)
            self.NextStates.pop(0)
            self.Dones.pop(0)
        self.States.append(state)
        self.Rewards.append(reward)
        self.Actions.append(action)
        self.NextStates.append(next_state)
        self.Dones.append(done)



    def run_episodes(self,env,episodes=1,e=0.1):
        for e in range(episodes):
            state,_ = env.reset()
            done= False
            turncated= False
            total_reward= 0
            while not done and not turncated:
                if np.random.rand()  >  e:
                    action  = self.act_train(env, state, use_random=False)
                else:
                    action = self.act_train(state,env, use_random=True)
                next_state, reward, done, turncated, _ = env.step(action)
                self.append_samples(state, action, reward,next_state, done)
                total_reward += reward
                state = next_state
            print(f"Episode reward:{total_reward}" )
            self.episode_rewards.append(total_reward)

    def initialize_buffer(self, envoriments):
        for env in envoriments:
            s, _ = env.reset()
            for _ in range(self.buffer_size // len(envoriments)):
                a = env.action_space.sample()
                s2, r, done, trunc, _ = env.step(a)
                self.append_samples(s, a, r, s2, done)
                if done or trunc:
                    s, _ = env.reset()
                else:
                    s = s2

    def collect_and_train(self, envoriments, iteration=100):
        # initialize buffer with random policy
        self.initialize_buffer(envoriments)
        
        # print("Buffer size ", len(self.States))

        for iter in range(iteration):
            S, A, R, S2, D = self.get_samples(iter)
            nb_samples = S.shape[0]
            SA = np.append(S, A, axis=1)

            if iter == 0:
                value = R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = self.Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2

            Q = RandomForestRegressor(n_estimators=50, random_state=42)
            Q.fit(SA, value)
            if iter > 0:
                self.residuals.append(np.mean((Q.predict(SA) - self.Qfunction.predict(SA))**2))

            self.Qfunction = Q

            for env in envoriments:
                self.run_episodes(env, e=0.1)

        np.savetxt("episode_rewards.csv", self.episode_rewards, delimiter=",", header="reward")
        np.savetxt("train_residuals.csv", self.residuals, delimiter=",", header="residual")

    def get_samples(self,iter):
        states = np.array(self.States)
        actions = np.array(self.Actions).reshape((-1, 1))
        rewards = np.array(self.Rewards)
        next_states = np.array(self.NextStates)
        dones = np.array(self.Dones)
        if iter == 0:
            # Combine all arrays into a single dataset
            dataset = list(zip(states, actions, rewards, next_states, dones))
            
            # Shuffle the combined dataset
            np.random.shuffle(dataset)
            
            # Unzip the shuffled dataset
            shuffled_states, shuffled_actions, shuffled_rewards, shuffled_next_states, shuffled_dones = zip(*dataset)
            
            return (
                np.array(shuffled_states),
                np.array(shuffled_actions).reshape((-1, 1)),
                np.array(shuffled_rewards),
                np.array(shuffled_next_states),
                np.array(shuffled_dones),
            )
        else:
            return (
                np.array(states),
                np.array(actions).reshape((-1, 1)),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
            )



if __name__ == "__main__":
    seed_everything()
    envoriments = create_envs(num_envs=2)
    agent = ProjectAgent()
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    agent.collect_and_train(envoriments, iteration=400)
    agent.save(path="compressed_model.pkl")
    end_time = datetime.now()
    print(f"Training ended at: {end_time}")
    print(f"Total training time: {end_time - start_time}")