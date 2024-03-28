# -*- coding: utf-8 -*-
"""
Created on Tue Oct  25 10:12:35 2022

@author: Ihtesham Shah
"""

import random
import numpy as np
from collections import deque
import sys
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
from TuristicFlowsEnv import Env
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from TouristicFlowsParameters import Param



env = Env()

state_size = env.observation_space.shape[0]

action_size = env.action_space.n

state_size = state_size
action_size = action_size

memory_MAXLEN = 1000000
memory = deque(maxlen=memory_MAXLEN)

gamma = 0.999              # discount rate
epsilon = 1.0              # exploration rate
epsilon_min = 0.001
epsilon_decay = 0.9999
learning_rate = 0.00001 
Tests = 10
batch_size = 64

PoIs = [
    "Museo",
    "S. Chiara",
    "C. monte",
    "Zona S. Carlo",
    "Riviera",
    "Posillipo",
    "Ercolano",
    "Vesuvio",
    "Pompei",
    "Positano",
    "Amalfi",
     "Sorrento",
    "Caserta Reggia",
    "Caserta Vecchia"
    ]

def _build_model( ):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=learning_rate))
   
    return model

def act(state, training = True):
    if training:
        if np.random.rand() <= epsilon:
            # Choose randmly to move in a PoI never visited before
            found = False
        
            while not found:
            
                next_action = random.randrange(action_size)
            
                if(env.current_state.get_visits()[next_action-1] == 0):
                    found = True                
        
            return next_action
        else:
            act_values = agent.predict(np.reshape(state.get_status(), [1, state_size]))

            return np.argmax(act_values[0])  # returns action
    else:
        act_values = agent.predict(np.reshape(state.get_status(), [1, state_size]))

        return np.argmax(act_values[0])  # returns action

def Index_value(arr, arr_size): # return the index value of the V(i)=k ; where k =[1,2,3....]
    first = max(arr) 
    
    second = -sys.maxsize
    for i in range(0, arr_size):
        if (arr[i] > second and
            arr[i] < first):
            second = arr[i]
 
    # Find third
    # largest element
    third = -sys.maxsize
    for i in range(0, arr_size):
        if (arr[i] > third and
            arr[i] < second):
            third = arr[i]
            
    if first == 3 :
 
        return [arr.index(first), arr.index(second), arr.index(third)]
    
    if first == 2:
        
        return [arr.index(first), arr.index(second)]


agent = _build_model()
agent.load_weights('ModelV0.11')
    
def main():  

    """
    Validation starts here
    """
    t_validation_rewards = list()
    x_validation = list()
    
    # Number of times the Cruise Shipe is delayed
    delays = 0          
    
    # Numeber of times the tourists are brought in an overbooked PoI
    overbookings = 0
    
    for e in range(Tests):
        state = env.reset()
    
        state = np.reshape(state, [1, state_size])
        
        time = 0
        treward = 0
        done = False
        
        while not done:
            action = act(env.current_state, training = False)
        
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
                      .format(e, Tests, treward, visits, delays, overbookings, env.Tmax-time))
                
                t_validation_rewards.append("test: {}/{}, score: {}, v: {}, delays: {}, overbookings: {}, residual time: {}"
                     .format(e, Tests, treward, visits, delays, overbookings, env.Tmax-time))
                
                
                break
    Values=visits
    arr_size= len(Values)
    #print("array", Values)
    Locations = Index_value(Values, arr_size)
    #print("First Location :", first_location, "Second Location:", second_location, "Third Location:", third_location )
    if len(Locations) == 2:
        Locations_to_visit = [PoIs[Locations[0]], PoIs[Locations[1]]]
 
    else: 
        Locations_to_visit = [PoIs[Locations[0]], PoIs[Locations[1]], PoIs[Locations[2]]]
    #print("First Location :", Locations_to_visit )
    
    return Locations_to_visit 

if __name__ == "__main__":
    main ()


