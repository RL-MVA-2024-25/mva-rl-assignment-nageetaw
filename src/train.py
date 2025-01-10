from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import joblib
from datetime import datetime
import torch
import random
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


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
        self.episode_rewards=[]
        self.Actions=[]
        self.States=[]
        self.NextStates=[]
        self.Rewards=[]
        self.Dones=[]
        self.Qfunction= None
        self.gamma=0.98
        self.nb_actions = env.action_space.n
        self.nb_states= env.observation_space.shape[0]
        self.residuals=[]
        self.buffer_size=50000
        # self.actions_during_train=[]

    def act(self, observation, use_random=False):
        # env.step() taken action index, so agent.act() must return action index 
        if use_random:
            action = env.action_space.sample()
        else:
            Qsa=[]
            for a in range(self.nb_actions):
                sa = np.append(observation,a).reshape(1, -1)
                Qsa.append(self.Qfunction.predict(sa)[0])
            action = np.argmax(Qsa)
            # print("action ",action)
        return action

    def save(self, path):
        joblib.dump(self.Qfunction, path)

    def load(self):
        self.Qfunction = joblib.load("model.pkl")

    def append_samples(self, state, action, reward, next_state, done):
        if len(self.States) >= self.buffer_size:
            # Remove the first (oldest) element from each list
            self.States.pop(0)
            self.Rewards.pop(0)
            self.Actions.pop(0)
            self.NextStates.pop(0)
            self.Dones.pop(0)
        
        # Add new samples to the lists
        self.States.append(state)
        self.Rewards.append(reward)
        self.Actions.append(action)
        self.NextStates.append(next_state)
        self.Dones.append(done)

        


    def run_episodes(self,env,episodes=1,e=0.1,disable_tqdm=False):
        for e in tqdm(range(episodes),disable=disable_tqdm):
            state,_ = env.reset()
            done= False
            turncated= False
            total_reward= 0
            while not done and not turncated:
                if np.random.rand()  >  e:
                    action  = self.act(state, use_random=False)
                else:
                    action = self.act(state, use_random=True)
                next_state, reward, done, turncated, _ = env.step(action)
                self.append_samples(state, action, reward,next_state, done)
                total_reward += reward
                state = next_state
            print(f"Episode reward:{total_reward}, buffer size:{len(self.States)}" )
            self.episode_rewards.append(total_reward)

    def get_samples(self):
        # state, action, reward,next_state, done
        return np.array(self.States), np.array(self.Actions).reshape((-1,1)) , np.array(self.Rewards), np.array(self.NextStates), np.array(self.Dones)
    
    
    def initialize_buffer(self,env,disable_tqdm=False):
        s, _ = env.reset()
        for _ in tqdm(range(self.buffer_size), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            self.append_samples(s,a,r,s2,done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2

           
        
    def collect_and_train(self,env,iteration= 1000,disable_tqdm=False):
        # initialize buffer with random policy
        self.initialize_buffer(env)
        print("Buffer size ",len(self.States))
        
        for iter in tqdm(range(iteration), disable=disable_tqdm):
            S,A,R,S2,D = self.get_samples()
            nb_samples = S.shape[0]
            SA = np.append(S,A,axis=1)

            if iter == 0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                # belman update
                value = R + self.gamma*(1-D)*max_Q2
            Q = RandomForestRegressor(n_estimators=50,random_state=42)
            Q.fit(SA,value)
            if iter > 0: 
                self.residuals.append(np.mean((Q.predict(SA)- self.Qfunction.predict(SA))**2))
            # policy update
            self.Qfunction = Q
            # run an epidoe with new policy and stack the buffer, pop out old samples
            self.run_episodes(env,episodes=1,e=0.1,disable_tqdm=True)

        # save residuals and rewards
        np.savetxt("episode_rewards.csv", self.episode_rewards, delimiter=",", header="reward")
        np.savetxt("train_residuals.csv", self.residuals, delimiter=",", header="residual")
    
            



if __name__ == "__main__":
    seed_everything()
    agent= ProjectAgent()
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    agent.collect_and_train(env,iteration=400)
    agent.save(path="model.pkl")
    end_time = datetime.now()
    print(f"Training ended at: {end_time}")
    print(f"Total training time: {end_time - start_time}")
