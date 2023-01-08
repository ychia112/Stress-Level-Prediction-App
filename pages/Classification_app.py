# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:59:53 2023

@author: jacky
"""
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from st_pages import Page, show_pages, add_page_title

add_page_title()
st.write('---')
st.write("""#### Answer the following questions:""")

# load data
stressData=pd.read_csv("stressData.csv")
stressData = stressData.drop([10005])
stressData.drop(['Timestamp'], axis=1, inplace=True)


def user_input():
    FRUITS_VEGGIES = st.slider('HOW MANY FRUITS OR VEGETABLES DO YOU EAT EVERYDAY?', 0, 5)
    PLACES_VISITED = st.slider('HOW MANY NEW PLACES DO YOU VISIT?', 0, 10)
    CORE_CIRCLE = st.slider('HOW MANY PEOPLE ARE VERY CLOSE TO YOU?', 0, 10)
    SUPPORTING_OTHERS = st.slider('HOW MANY PEOPLE DO YOU HELP ACHIEVE A BETTER LIFE?', 0, 10)
    SOCIAL_NETWORK = st.slider('WITH HOW MANY PEOPLE DO YOU INTERACT WITH DURING A TYPICAL DAY?', 0, 10)
    ACHIEVEMENT = st.slider('HOW MANY REMARKABLE ACHIEVEMENTS ARE YOU PROUD OF?', 0, 10)
    DONATION = st.slider('HOW MANY TIMES DO YOU DONATE YOUR TIME OR MONEY TO GOOD CAUSES?', 0, 5)
    BMI_RANGE = st.selectbox('WHAT IS YOUR BODY MASS INDEX (BMI) RANGE?', (0, 1))
    TODO_COMPLETED = st.slider('HOW WELL DO YOU COMPLETE YOUR WEEKLY TO-DO LISTS?', 0, 10)
    FLOW = st.slider('IN A TYPICAL DAY, HOW MANY HOURS DO YOU EXPERIENCE "FLOW"?', 0, 10)
    DAILY_STEPS = st.slider('HOW MANY STEPS (IN THOUSANDS) DO YOU TYPICALLY WALK EVERYDAY?', 1, 10)
    LIVE_VISION = st.slider('FOR HOW MANY YEARS AHEAD IS YOUR LIFE VISION VERY CLEAR FOR?', 0, 10)
    SLEEP_HOURS = st.slider('ABOUT HOW LONG DO YOU TYPICALLY SLEEP?', 1, 10)
    LOST_VACATION = st.slider('HOW MANY DAYS OF VACATION DO YOU TYPICALLY LOSE EVERY YEAR ?', 0, 10)
    DAILY_SHOUTING = st.slider('HOW OFTEN DO YOU SHOUT OR SULK AT SOMEBODY?', 0, 10)
    SUFFICIENT_INCOME = st.selectbox('HOW SUFFICIENT IS YOUR INCOME TO COVER BASIC LIFE EXPENSES?', (1, 2))
    PERSONAL_AWARDS = st.slider('HOW MANY RECOGNITIONS HAVE YOU RECEIVED IN YOUR LIFE?', 0, 10)
    TIME_FOR_PASSION = st.slider('HOW MANY HOURS DO YOU SPEND EVERYDAY DOING WHAT YOU ARE PASSIONATE ABOUT?', 0, 10)
    WEEKLY_MEDITATION = st.slider('IN A TYPICAL WEEK, HOW MANY TIMES DO YOU HAVE THE OPPORTUNITY TO THINK ABOUT YOURSELF?', 0, 10)
    AGE = st.selectbox('AGE GROUPS',("less than 20", "21 to 35", "36 to 50", "51 or more"))
    GENDER = st.selectbox('GENDER', ("Female", "Male"))
    ok = st.button("Done")
    input_data = {"FRUITS_VEGGIES":FRUITS_VEGGIES,
                    "PLACES_VISITED":PLACES_VISITED,
                    "CORE_CIRCLE":CORE_CIRCLE,
                    "SUPPORTING_OTHERS":SUPPORTING_OTHERS,
                    "SOCIAL_NETWORK":SOCIAL_NETWORK,
                    "ACHIEVEMENT":ACHIEVEMENT,
                    "DONATION":DONATION,
                    "BMI_RANGE":BMI_RANGE,
                    "TODO_COMPLETED":TODO_COMPLETED,
                    "FLOW":FLOW,
                    "DAILY_STEPS":DAILY_STEPS,
                    "LIVE_VISION":LIVE_VISION,
                    "SLEEP_HOURS":SLEEP_HOURS,
                    "LOST_VACATION":LOST_VACATION,
                    "DAILY_SHOUTING":DAILY_SHOUTING,
                    "SUFFICIENT_INCOME":SUFFICIENT_INCOME,
                    "PERSONAL_AWARDS":PERSONAL_AWARDS,
                    "TIME_FOR_PASSION":TIME_FOR_PASSION,
                    "WEEKLY_MEDITATION":WEEKLY_MEDITATION,
                    "AGE":AGE,
                    "GENDER":GENDER}
    features = pd.DataFrame(input_data, index=[0])
    return features, ok
#%% data pre-processing
X = stressData.drop("DAILY_STRESS", axis=1)
y = stressData["DAILY_STRESS"]
df, ok = user_input()

#encoder for input
if df['AGE'][0] == 'less than 20':
    df['AGE'][0] = 3
if df['AGE'][0] == '21 to 35':
    df['AGE'][0] = 0
if df['AGE'][0] == '36 to 50':
    df['AGE'][0] = 1
if df['AGE'][0] == '51 or more':
    df['AGE'][0] = 2
    
if df['GENDER'][0] == 'Female':
    df['GENDER'][0] = 0
if df['GENDER'][0] == 'Male':
    df['GENDER'][0] = 1
    
from sklearn.preprocessing import LabelEncoder

# encoder for stressData
le_age = LabelEncoder()
X['AGE'] = le_age.fit_transform(X['AGE'])
X['AGE'].unique()

le_gender = LabelEncoder()
X['GENDER'] = le_gender.fit_transform(X['GENDER'])
X['GENDER'].unique()

#%% model



























