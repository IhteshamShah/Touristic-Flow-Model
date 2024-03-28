# -*- coding: utf-8 -*-
"""
Created on Tue Oct  20 11:36:25 2022

@author: Ihtesham Shah
"""


import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import practice as PT
import pandas as pd


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

def  load_lottieurl (url): # to get the animations from address
    r = requests.get(url)
    if r.status_code != 200:
        return None 
    return r.json() 

#here are the addresses of animations
lottie_coding =  load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_mdbdc5l7.json")
lottie_coding2= load_lottieurl("https://assets1.lottiefiles.com/datafiles/AtGF4p7zA8LpP2R/data.json")
lottie_coding3 =  load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_bhebjzpu.json")

#top container 
with st.container():   
    left_column,center_column, right_column = st.columns(3)
    with left_column:
        st.subheader(" ")
        st_lottie(lottie_coding, height=150)

    with center_column:
        st.subheader("Welcome to Touristic Flow, :wave:")
        st_lottie(lottie_coding2, height=150)
        
    with right_column:
        st.subheader(" ")
        st_lottie(lottie_coding3, height=150)
        
with st.container():
    st.write("---")

result=PT.main() #run the practice file and store the POIs to be visited

def main(result):
    with st.container():

            st.title("ICAR Touristic Flow Model")
            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Deep reinforcment learning model </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)
            CS = st.selectbox("Cruise Ship state",options=['select', "arrived", "not arrived"])
            nTourists = st.selectbox("Number of Tourist", options=['select',  20, 59, 60])
            PoI_MAX_Capacity = st.selectbox("Maximum number of people admitable", options=['select',0, 10, 20, 30, 40, 50, 60])
            CRC = st.selectbox("Current residual Capacity", options=['select',0, 10, 20, 30, 40, 50, 60])
                      
            mylist1= [20, 59, 60]
            mylist2= [0, 10, 20, 30, 40, 50, 60]
            
            if st.button("Predict"): 
                
                if CS != "arrived":
                    st.error("ship not arrived")
                    
                elif nTourists not in mylist1:
                    st.error("Invalid Number of Tourist")
                elif PoI_MAX_Capacity not in mylist2 :
                    st.error("Invalid PoI_MAX_Capacity")
                elif CRC not in mylist2 :
                    st.error("Current residual Capacity for an PoI")
                    
                 
                    
                
                else:
                    
                    st.write("Here are the list of places to visit during the same tour")
                    
                
                    if len(result)==3:
                
                        st.success(result[2])
                        st.success(result[1])
                        st.success(result[0])
                        
                    if len(result) ==2:
                        st.success(result[1])
                        st.success(result[0])
                        
                    
                
    with st.container():
            st.write("---")
            if st.button("About"):
                st.write(" ")
                
            st.write("""This work present a Deep Reinforcement Learning based 
                     planner for the onshore touristic itineraries and the intelligent 
                     distribution of cruise passengers in a city. The aim is to maximise 
                     the number of touristic attraction locations visited during a tour 
                     by avoiding the overcrowding of the touristic attraction locations. 
                     The planner is able to compose onshore touristic itineraries 
                     meeting the constraint given within the time window available 
                     for the cruise passengers, by taking into account the number of 
                     attraction locations available in the city together with dynamic 
                     parameters as their reception capacity and the time needed to 
                     go from one attraction to another""")
            st.write(" [Learn more ] < ( https://ieeexplore.ieee.org/document/9486648 )")
                
                
                
                

if __name__=='__main__':
    main(result)
