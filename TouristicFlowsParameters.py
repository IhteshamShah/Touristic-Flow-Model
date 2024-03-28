#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:49:30 2020

@author: Antonio Coronato
"""

import numpy as np

class Param:
    def __init__(self):
            """
            Env Model V0.1

            Variables:
            Tmax [Minutes] = Time the cruise ship stays in the port
            N = Number of PoI
            M = Number of buses
            nTourists = Number of tourists
            CRC(i) = Current residual Capacity for i-th PoI
            Distances(i,j) = Distance between i-th PoI and j-th PoI
            Bpos(i) = Positioin of i-th bus
            Bstatus(i) = Status of -th bus
            TvF(i) = Time for a visit at i-th PoI
            V(i) = Visit of PoI_i => V(i) = 0 means PoI_i has not been visited yet
                                  => V(i) = k means PoI_i has been the k-th PoI visited during the same tour
            CS = Cruise Ship state

            Aliases
            D(i,j) = Distances(i,j)

            OBSERVATION SPACE
            Time, CRC(PoI_1),..., CRC(PoI_N), ntourists, D(0,0),...,D(0,N),...,D(N,0),...,D(N,N),
                Bpos(1),...,Bpos(M),Bstatus(1),....,Bstatus(M), TfV(PoI_1),...TfV(PoI_N), V(PoI_1),..., V(PoI_N), CS

            Bus Position:
            Pos = -1 -> Bus in transit
            Pos = 0 -> Port
            Pos = 1 -> PoI_1
            Pos = 2 -> PoI_2
            ...
            Pos = N -> PoI_N

            Bus status
            s = 0 -> available (ready to go in a new position)
            s = 1 -> not available (already moving or waiting people in a PoI)

            Cruise Ship status
            s = 0 -> Ship at the port
            s = 1 -> Ship has left the port

            ACTION SPACE  *** NEW in this version
            DIM = (N+1)

            Actions
            action = 0 -> move to 0
            action = 1 -> move to 1
            ...
            action = k -> move to k
            

            HYPOTHESYS
            M = 1 -> Only one virtual bus. This allows the agent to send all real buses along the same route
            Distances (in minutes) between two positions are constant and we do not care about traffic conditions
            Time for a visit of a Point of Interest is constant


            :param Tmax: Maximium time steps for an episode (Maximum time the cruise ship waits for passengers)
            :param MINnumber_of_tourists: Minimum number of tourists from the cruise ship
            :param MAXnumber_of_tourists: Maximum number of tourists from the cruise ship
            :param PoI_MAX_Capacity: Maximum number of people admitable by the i-th PoI
            :param PoI_Time_for_Visit: Time for a visit at the i-th PoI
            :param Distances: Distances between PoIs
            """

            # Max number of minutes the cruise ship will wait at the port
            self.Tmax = 480         
            
            self.MAXTourists = 60
            self.MINTourists = 59

            # Points of Interest
            self.PoIs = {#0 : "Porto", 
                1 : "Museo",
                2 : "S. Chiara",
                3 : "C. monte",
                4 : "Zona S. Carlo",
                5 : "Riviera",
                6 : "Posillipo",
                7 : "Ercolano",
                8 : "Vesuvio",
                9 : "Pompei",
                10 : "Positano",
                11 : "Amalfi",
                12 : "Sorrento",
                13 : "Caserta Reggia",
                14 : "Caserta Vecchia"
                }

            # Points of Interest max capacity [# of People]
            self.__PoI_Initial_Capacity = [120, 120, 90, 120, 90, 90, 240, 240, 240, 240, 240, 240, 240, 150]
            
            # Time needed to visit a Points of Interest [Minutes]
            self.__PoI_Time_for_Visit = [120, 30, 120, 120, 30, 30, 180, 120, 180, 90, 90, 90, 120, 90]
    
            # Mean temporal distance values [minutes] between two PoIs
            self.__Initial_Distances = [[1, 26, 11, 29, 9, 18, 25, 27, 31, 34, 94, 95, 84, 55, 67],
                                         [25, 1, 20, 8, 17, 19, 26, 30, 29, 33, 91, 93, 65, 39, 52],
                                         [21, 5, 1, 12, 19, 24, 29, 30, 33, 36, 92, 96, 73, 41, 54] ,
                                         [23, 10, 22, 1, 22, 19, 20, 20, 24, 27, 83, 86, 73, 38, 51],
                                         [7, 20, 8, 26, 1, 15, 22, 25, 38, 40, 95, 99, 81, 50, 60],
                                         [6, 25, 15, 27, 15, 1, 12, 28, 39, 46, 93, 95, 81, 53, 62],
                                         [13, 31, 22, 28, 21, 8, 1, 38, 43, 45, 102, 102, 88, 61, 72],
                                         [31, 28, 42, 24, 27, 34, 37, 1, 28, 16, 72, 73, 57, 41, 53],
                                         [36, 32, 49, 28, 31, 41, 42, 28, 1, 34, 73, 77, 54, 39, 52],
                                         [42, 38, 50, 35, 41, 47, 54, 21, 34, 1, 68, 73, 46, 50, 62],
                                         [99, 89, 95, 87, 93, 102, 108, 73, 71, 64, 1, 41, 39, 102, 113],
                                         [97, 87, 97, 89, 91, 102, 108, 73, 73, 61, 45, 1, 69, 102, 114],
                                         [85, 74, 77, 93, 77, 86, 93, 59, 53, 41, 37, 74, 1, 86, 98],
                                         [54, 39, 53, 41, 50, 53, 53, 36, 37, 43, 99, 99, 85, 1, 23],
                                         [67, 54, 69, 53, 62, 65, 67, 51, 52, 56, 116, 114, 101, 23, 1]
                         ]

            self.__number_of_points = len(self.__PoI_Initial_Capacity)
            
            # (b - a) of the uniform distribution for PoIs current residual capacity
            self.__VAR_CRC = 0.1

            # standard deviation of the normal distribution for the temporal distance betweeen two PoIs
            self.__scale = 2
            
            self.reset()


    def reset(self):
            self.__PoI_Current_Capacity = self.__PoI_Initial_Capacity[:]
            
            self.Current_Distances = []
            
            #self.Current_Distances = self.__Initial_Distances[::]
            for i in range(len(self.__Initial_Distances)):
                new = []
                for j in range(len(self.__Initial_Distances[i])):
                    #self.Current_Distances[i][j] = self.__Initial_Distances[i][j]
                    new.append(self.__Initial_Distances[i][j])
                    
                self.Current_Distances.append(new)
                    
            # Number of tourists
            self.__number_of_tourists = np.random.randint(self.MINTourists, self.MAXTourists)
    
    
    def get_number_of_tourists(self, update = True):
        if update:
            self.__number_of_tourists = np.random.randint(self.MINTourists, self.MAXTourists)
            
        return self.__number_of_tourists

    
    def get_PoIs(self):
        return self.PoIs
    
   
    def get_Tmax(self):
        return self.Tmax
    

    def get_number_of_points(self):
        return self.__number_of_points


    def get_PoI_Current_Capacity(self, update = True):        
        if update:
            # Update the current Capacity
            for i in range(self.__number_of_points):
                self.__PoI_Current_Capacity[i] = int(self.__PoI_Initial_Capacity[i]*np.random.uniform(low = 1 -self.__VAR_CRC, high = 1 + self.__VAR_CRC))

                if (self.__PoI_Current_Capacity[i] <= 0): 
                    self.__PoI_Current_Capacity[i] = self.__PoI_Initial_Capacity[i]

        return self.__PoI_Current_Capacity


    def get_Current_Distances(self, update = True):
        if update:
                for i in range(len(self.__Initial_Distances)):
                    for j in range(len(self.__Initial_Distances[i])):
                        if (i != j):
                            self.Current_Distances[i][j] = int(np.random.normal(loc = self.__Initial_Distances[i][j], scale=self.__scale))
        
        return self.Current_Distances

    def get_PoI_Time_for_Visit(self):
        return self.__PoI_Time_for_Visit
    
    def get_distance(self, current_pos, dest):
        return self.__Current_Distances[current_pos, dest]
    
