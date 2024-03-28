#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:59:09 2019
tijuana1


@author: antonio
"""

import random
import numpy as np
from collections import deque
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
from TuristicFlowsEnv import Env
import matplotlib.pyplot as plt
import statistics as st
import csv 
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


EPISODES = 15
TESTS = 10

# writing to csv file 
FIRST_TIME = False

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory_MAXLEN = 1000000
        self.memory = deque(maxlen=self.memory_MAXLEN)
        
        self.gamma = 0.999              # discount rate
        self.epsilon = 1.0              # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.00001 

        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
       
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training = True):
        if training:
            if np.random.rand() <= self.epsilon:
                # Choose randmly to move in a PoI never visited before
                found = False
            
                while not found:
                
                    next_action = random.randrange(self.action_size)
                
                    if(env.current_state.get_visits()[next_action-1] == 0):
                        found = True                
            
                return next_action
            else:
                act_values = self.model.predict(np.reshape(state.get_status(), [1, state_size]))

                return np.argmax(act_values[0])  # returns action
        else:
            act_values = self.model.predict(np.reshape(state.get_status(), [1, state_size]))

            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = Env()
    
    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

 #   batch_size = 32
    batch_size = 64
    
    """
        Training starts here
    """
    trewards = list()
    x = list()

    startTime = datetime.datetime.now()

    for e in range(0,EPISODES):
        state = env.reset()

        state = np.reshape(state, [1, state_size])

        time = 0
        treward = 0
        done = False

        while not done:
            action = agent.act(env.current_state)

            nextState, reward, done = env.step(action)

            """
                Start here to get state variables
            """
            time = nextState.get_time()
            crc = nextState.get_crc()
            nTourists = nextState.get_nTourists()
            bPos = nextState.get_bPos()
            bStatus = nextState.get_bStatus()
            visits = nextState.get_visits()
            cs = nextState.get_cs()
            """
                Stop here to get state variables
            """
            
            treward += reward

            newState = np.reshape(nextState.get_status(), [1, state_size])
            agent.remember(state, action, reward, newState, done)
            state = newState

            if done:
                print("episod: {}/{}, score: {}, e: {:.2}, v: {}, cs: {}"
                      .format(e, EPISODES, treward, agent.epsilon, visits, cs))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        trewards.append(treward);
        x.append(e);

    """
        Training stops here
    """

    # Save the model
    agent.save("ModelV1.0")

    """
        Validation starts here
    """
    t_validation_rewards = list()
    x_validation = list()
    
    # Number of times the Cruise Shipe is delayed
    delays = 0          
    
    # Numeber of times the tourists are brought in an overbooked PoI
    overbookings = 0

    for e in range(0,TESTS):
        state = env.reset()

        state = np.reshape(state, [1, state_size])

        time = 0
        treward = 0
        done = False

        while not done:
            action = agent.act(env.current_state, training = False)

            nextState, reward, done = env.step(action)

            """
                Start here to get state variables
            """
            time = nextState.get_time()
            crc = nextState.get_crc()
            nTourists = nextState.get_nTourists()
            bPos = nextState.get_bPos()
            bStatus = nextState.get_bStatus()
            visits = nextState.get_visits()
            cs = nextState.get_cs()
            """
                Stop here to get state variables
            """
            
            treward += reward

            newState = np.reshape(nextState.get_status(), [1, state_size])
            #agent.remember(state, action, reward, newState, done)
            state = newState

            if done:
                if (cs == 1):
                    delays += 1
                    
                if (cs == 0) and (reward < 0):
                    overbookings += 1
                    
                print("test: {}/{}, score: {}, v: {}, delays: {}, overbookings: {}, residual time: {}"
                      .format(e, TESTS, treward, visits, delays, overbookings, env.Tmax-time))

                break

            #if len(agent.memory) > batch_size:
            #    agent.replay(batch_size)
        
        t_validation_rewards.append(treward);
        x_validation.append(e);

    """
        Validation stops here
    """




    """
        Store results here
    """
    # field names 
    fields = ['Episodes', 'Mean reward', 'MAX Rew', 'MIN Rew', 'MAXTurists', 'MINTurists', 'nTourists', 'Final Epsilon', 'Epsilon_decay', 'Learning_rate','Gamma','BATCH_SIZE','MAXLEN','Stat','Stop','Training time','Model name',
              'Tests', 'Mean _validation_reward', 'MAX Val Rew', 'MIN Val Rew', 'Delays', 'Overbookings']

    stopTime = datetime.datetime.now()
    trainingTime = stopTime - startTime
    

    # data rows of csv file
    row = [EPISODES, st.mean(trewards), max(trewards), min(trewards), env.MAXnumber_of_tourists, env.MAXnumber_of_tourists, env.nTourists, agent.epsilon, agent.epsilon_decay, agent.gamma, agent.learning_rate, batch_size, agent.memory_MAXLEN, startTime, stopTime, trainingTime, 'ModelV0.11',
           TESTS, st.mean(t_validation_rewards), max(t_validation_rewards), min(t_validation_rewards), delays, overbookings]
      
    # name of csv file 
    filename = "results.csv"
  
    if FIRST_TIME:
        with open(filename, 'w') as csvfile:
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
          
            # writing the fields 
            csvwriter.writerow(fields)
          
            # writing the data rows 
            csvwriter.writerow(row)
    else:
        with open(filename, 'a') as csvfile:
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
          
            # writing the data rows 
            csvwriter.writerow(row)

    
    print("Training EPISODES: ",EPISODES)
    print("Training mean reward: ",st.mean(trewards))
    print("Training MAX reward: ",max(trewards))
    print("Training MIN reward: ",min(trewards))
    
    #plt.plot(x, trewards, label='Epsilon decay = '+str(agent.epsilon_decay))
    plt.plot(x, trewards)
    plt.title("Training - Total reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    #plt.legend()
    plt.show()

    print("Validation TESTS: ",TESTS)
    print("Validation mean reward: ",st.mean(t_validation_rewards))
    print("Validation MAX reward: ",max(t_validation_rewards))
    print("Validation MIN reward: ",min(t_validation_rewards))
    
    #plt.plot(x, trewards, label='Epsilon decay = '+str(agent.epsilon_decay))
    plt.plot(x_validation, t_validation_rewards)
    plt.title("Validation - Total reward")
    plt.xlabel("Test")
    plt.ylabel("Reward")
    #plt.legend()
    plt.show()
    
