#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:09:26 2020

@author: Antonio Coronato

    THis program generates a dataset of TESTS for the validation of the model
"""

#import numpy as np
import csv 
from TouristicFlowsParameters import Param

TESTS = 100 # Number of tests for the Test Set

if __name__ == "__main__":
    param = Param()
    
    status_size = 0
   
    status = list()
    fields = list()
       
    FIRST = True

    with open('testSet.csv', 'w') as writeFile:
        writer = csv.writer(writeFile, delimiter=',')

        for i in range(TESTS):
            # Set time variable to 0
            fields.append('Time')
            status.append(0)
            status_size = 1
            
            # Set the number of tourists        
            fields.append('nTourists')
            number_of_tourists = param.get_number_of_tourists()
            status.append(number_of_tourists)
            status_size += 1
        
            # Set the initial capacity for the PoIs
            crc = param.get_PoI_Current_Capacity()
        
            for i in range(len(crc)):
                str = f"crc{[i]}"
                fields.append(str)
                
            status.extend(crc)
            status_size += len(crc)
            
            #crc.clear()
            
            distances = param.get_Current_Distances()
            
            #count = 0
            for i in range(len(distances)):
                for j in range(len(distances[i])):
                    str = f"d[{i}][{j}]"
                    fields.append(str)
                    status.append(distances[i][j])
                    #count += 1

            status_size += len(distances)*len(distances)
            
            #distances.clear()

            # Set the initial bus position to 0
            fields.append('bPos')
            status.append(0)
            status_size += 1

            # Set the initial bus status to 0 (available)
            fields.append('bStatus')
            status.append(0)
            status_size += 1
        
            # Set the initial set of visits performed
            for i in range (param.get_number_of_points()):
                str = f"V[{i}]"
                fields.append(str)
                status.append(0)
                status_size += 1
            
            # Set the Cruise Ship status to 0 (waiting at the port)
            fields.append('CruiseShipStatus')
            status.append(0)
            status_size += 1

            if FIRST:
                writer.writerow(fields)
                FIRST = False
        
            writer.writerow(status)

            status.clear()
     
    writeFile.close()
    