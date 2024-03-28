#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:44:22 2019

@author: Antonio Coronato
"""


#import math
import gym
import numpy as np
from TouristicFlowsParameters import Param
from gym import spaces

class Env():
    def __init__(self):
        self.param = Param()
        
        self.Tmax = self.param.get_Tmax()
        
        self.nPoints = self.param.get_number_of_points()

        self.MAXnumber_of_tourists = self.param.MAXTourists
        self.MINnumber_of_tourists = self.param.MINTourists

        # Maximum capcacity (in terms of people) and time needed for a visit for for the Points of Interest
        self.PoI_Capacity = self.param.get_PoI_Current_Capacity(update = False)
        self.PoI_Time_for_Visit = self.param.get_PoI_Time_for_Visit()
        

        # Mean distances between Points of Interest (including the port)
        self.Distances = self.param.get_Current_Distances(update = False)

        #self.bus = Bus()


        """
            *** Here we start to define the observation space
        """

        min = list()
        max = list()

        # Time variable
        min.append(0)
        max.append(self.Tmax)

        # Actual number of tourists
        min.append(self.MINnumber_of_tourists)
        max.append(self.MAXnumber_of_tourists)

        # Current Residual Capacity for the i-th PoI
        for i in range(self.nPoints):
            min.append(0)
            max.append(self.PoI_Capacity[i])

        # Temporal distances between two PoIs
        for i in range(self.nPoints+1):
            for j in range(self.nPoints+1):
                min.append(self.Distances[i][j])
                max.append(self.Distances[i][j])

        # Column N+2 for the bus position
        min.append(-1)
        max.append(self.nPoints)

        # Column N+3 for the bus status
        min.append(0)
        max.append(1)

        # Visits 
        for i in range(self.nPoints):
            min.append(0)
            max.append(self.nPoints)

        # Cruise Ship Status
        min.append(0)
        max.append(1)

        low_state = np.array(min)
        high_state = np.array(max)

        self.observation_space = spaces.Box(low=low_state, high=high_state)       
        """
            *** Here we finish to define the observation space
        """


        """
            *** Here we start to define the activity space
        """
        self.action_space = spaces.Discrete(self.nPoints + 1)
        """
            action = 0 -> move to 0
            action = 1 -> move to 1
            ...
            action = k -> move to k 
        """

        """
            *** Here we set the initial state
        """
        self.reset()

    def reset(self):
        """

        :return:  np.array(self.current_state)
        """

        self.param.reset()
        
        """
           *** Here we start to build the initial status of the environment
        """
        self.visits = [0] * self.nPoints

        self.current_state = EnvState(self.param.get_number_of_tourists(), 
                                      self.param.get_PoI_Current_Capacity(), 
                                      self.param.get_Current_Distances(), 
                                      self.visits)
        
        #print("CURRENT STATE: ",self.current_state.get_status())
        """
            *** Here we finish to build the initial status of the environment
        """

        return np.array(self.current_state.get_status())

    # take an action
    def step(self, action):
        
        """
            Start here to get the current state
        """
        time = self.current_state.get_time()
        crc = self.current_state.get_crc()
        #print("STATE ACT: ",crc)
        self.nTourists = self.current_state.get_nTourists()
        bPos = self.current_state.get_bPos()
        bStatus = self.current_state.get_bStatus()
        visits = self.current_state.get_visits()
        cs = self.current_state.get_cs()
        """
            Stop here to get the current state
        """

        """
            Decode the action here
        """
        startPos = bPos
        destPos = action 

        """
            Start here to build the next state
        """
        if startPos > 0:
            # Increase capability for the starting position, which is not the port
            crc[startPos - 1] += self.nTourists

        if destPos > 0:
            # Decrease the capability for the destination, which is not the port
            crc[destPos - 1] -= self.nTourists

            if crc[destPos - 1] < 0 or visits[destPos - 1] > 0:
                # Destination unable to accept the amount of tourists or already visited
                done = True
                reward = -10

            else:
                visits[destPos - 1] = visits[np.argmax(visits)] + 1
                done = False
                reward = 1
        else:
            done = True
            reward = 0

        # Set current time to arrival time
        time += self.Distances[startPos][destPos]
        time += self.PoI_Time_for_Visit[destPos - 1]

        if time > self.Tmax:
            # Cruise ship has left!
            cs = 1
            done = True
            reward = -100

        next_state = EnvState(self.nTourists, crc, self.Distances, visits, time=time, bPos=destPos, bStatus=bStatus, cs=cs)

        self.current_state = next_state

        return self.current_state, reward, done

    def __getDistanceBetween(self,startPos,destPos):
        index = 2 + self.nPoints + (self.nPoints + 1)*startPos + destPos
        distance = self.Distances[startPos][destPos]
        return index, distance



class EnvState():
    def __init__(self, nTourists, crc, distances, visits, time = 0, bPos = 0, bStatus = 0, cs = 0):
        self.time = time
        self.nTourists = nTourists
        self.crc = crc

        self.distances = list()
        
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                self.distances.append(distances[i][j])

        self.bPos = bPos
        self.bStatus = bStatus
        self.visits = visits
        self.cs = cs


    def get_status(self):
        status = list()
        
        status.append(self.time)
        status.append(self.nTourists)
        status.extend(self.crc)
        status.extend(self.distances)
        status.append(self.bPos)
        status.append(self.bStatus)
        status.extend(self.visits)
        status.append(self.cs)
        
        return status
      
                
    def get_nTourists(self):
        return self.nTourists
    

    def get_time(self):
        return self.time
    
    
    def get_bPos(self):
        return self.bPos


    def get_bStatus(self):
        return self.bStatus

    
    def get_cs(self):
        return self.cs
    

    def get_crc(self):
        return self.crc
    

    def get_visits(self):
        return self.visits
