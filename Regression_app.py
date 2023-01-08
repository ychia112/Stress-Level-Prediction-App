# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:59:53 2023

@author: jacky
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

st.write("# Daily Stress Prediction and Work Life Balance Grading")
st.write('---')
st.write("""#### Answer the following questions:""")

# load data
stressData=pd.read_csv("stressData.csv")
stressData = stressData.drop([10005])
stressData.reset_index(inplace=True)
stressData.set_index('index')
stressData.drop(['index'], axis=1, inplace=True)
stressData.drop(['Timestamp'], axis=1, inplace=True)
stressData['DAILY_STRESS'] = stressData['DAILY_STRESS'].astype(int)

stressData.loc[stressData['DAILY_STRESS'] == 0, 'DAILY_STRESS'] = 0
stressData.loc[stressData['DAILY_STRESS'] == 1, 'DAILY_STRESS'] = 0
stressData.loc[stressData['DAILY_STRESS'] == 2, 'DAILY_STRESS'] = 0
stressData.loc[stressData['DAILY_STRESS'] == 3, 'DAILY_STRESS'] = 1
stressData.loc[stressData['DAILY_STRESS'] == 4, 'DAILY_STRESS'] = 1
stressData.loc[stressData['DAILY_STRESS'] == 5, 'DAILY_STRESS'] = 1


filt1 = (stressData['DAILY_STRESS'] == 1)
output_1 = stressData.loc[filt1]               # data that output = 1
filt0 = (stressData['DAILY_STRESS'] == 0)
output_0 = stressData.loc[filt0]               # data that output = 0
output_1 = output_1.sample(n = len(output_0))     # set the ratio of output_1 and output_0
stressData = pd.concat([output_1, output_0]) 

def user_input():
    FRUITS_VEGGIES = st.slider('How many fruits or vegetables do you eat everyday?', 0, 5)
    PLACES_VISITED = st.slider('How many new places do you visit? (Over a period of 12 months.)', 0, 10)
    CORE_CIRCLE = st.slider('How many people are very close to you?', 0, 10)
    SUPPORTING_OTHERS = st.slider('How many people do you help achieve a better life? (Over a period of 12 months. e.g.: caring for your family, actively supporting a friend, mentoring, coaching,...)', 0, 10)
    SOCIAL_NETWORK = st.slider('With how many people do you interact with during a typical day?', 0, 10)
    ACHIEVEMENT = st.slider('How many remarkable achievements are you proud of? (Over the last 12 months)', 0, 10)
    DONATION = st.slider('How many times do you donate your time or money to good causes? (Over a period of 12 months. Include financial donation, your time contribution, fundraising, volunteering,...)', 0, 5)
    BMI_RANGE = st.selectbox('What is your body mass index (bmi) range? (<25:0 / >=25:1)', (0, 1))
    TODO_COMPLETED = st.slider('How well do you complete your weekly to-do lists?', 0, 10)
    FLOW = st.slider('In a typical day, how many hours do you experience "flow"? ( Hours per day. Flow is defined as the mental state, in which you are fully immersed in performing an activity.)', 0, 10)
    DAILY_STEPS = st.slider('How many steps (in thousands) do you typically walk everyday?', 1, 10)
    LIVE_VISION = st.slider('For how many years ahead is your life vision very clear for? (I do not have a life vision: 0/ Years or more: 10)', 0, 10)
    SLEEP_HOURS = st.slider('About how long do you typically sleep? (Hours, choose 10 if >10 hours)', 1, 10)
    LOST_VACATION = st.slider('How many days of vacation do you typically lose every year?', 0, 10)
    DAILY_SHOUTING = st.slider('How often do you shout or sulk at somebody? (Times per week. Expressing your negative emotions in an active or passive manner.)', 0, 10)
    SUFFICIENT_INCOME = st.selectbox('How sufficient is your income to cover basic life expenses? (Not sufficient:1 / Sufficient:2)', (1, 2))
    PERSONAL_AWARDS = st.slider('How many recognitions have you received in your life? ( E.g.: diploma, degree, certificate, accreditation, award, prize, published book, presentation at major conference, medals, cups, titles...)', 0, 10)
    TIME_FOR_PASSION = st.slider('How many hours do you spend everyday doing what you are passionate about? (Hours)', 0, 10)
    WEEKLY_MEDITATION = st.slider('In a typical week, how many times do you have the opportunity to think about yourself? ( Include meditation, praying and relaxation activities such as fitness, walking in a park or lunch breaks)', 0, 10)
    AGE = st.selectbox('Age groups',("less than 20", "21 to 35", "36 to 50", "51 or more"))
    GENDER = st.selectbox('Gender', ("Female", "Male"))
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
X = stressData.drop(["DAILY_STRESS"], axis=1)
X = X.drop(["WORK_LIFE_BALANCE_SCORE"], axis = 1)
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

df2 = df.copy()  

from sklearn.preprocessing import LabelEncoder

# encoder for stressData
le_age = LabelEncoder()
X['AGE'] = le_age.fit_transform(X['AGE'])
X['AGE'].unique()

le_gender = LabelEncoder()
X['GENDER'] = le_gender.fit_transform(X['GENDER'])
X['GENDER'].unique()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
df = scaler.transform(df)
#%% model
from sklearn.ensemble import AdaBoostClassifier
ADA_result = 0
if ok:  
    ADA = AdaBoostClassifier()
    ADA = ADA.fit(X, y)
    ADA_result = ADA.predict(df)
    if ADA_result == 1:
        result = 'high'
    else:
        result = 'low'
    st.subheader(f"You stress level is {result}.")



X = stressData.drop("WORK_LIFE_BALANCE_SCORE", axis=1)
y = stressData["WORK_LIFE_BALANCE_SCORE"]
    
from sklearn.preprocessing import LabelEncoder

# encoder for stressData
le_age = LabelEncoder()
X['AGE'] = le_age.fit_transform(X['AGE'])
X['AGE'].unique()

le_gender = LabelEncoder()
X['GENDER'] = le_gender.fit_transform(X['GENDER'])
X['GENDER'].unique()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
df2.insert(1,'DAILY_STRESS',[ADA_result])
df2 = scaler.transform(df2)
df2 = pd.DataFrame(df2)
#%% model
def linear_regression(x,y):
  x=np.concatenate([np.ones((x.shape[0],1)),x],axis=1)
  beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return beta

weights=linear_regression(np.array(X),np.array(y))
df2.insert(0, column="intercept", value=1)

prediction = sum(df2.iloc[0]*weights.T)

mean_y = np.mean(y)
std_y = np.std(y)
result = 'EXCELLENT'
if prediction < mean_y + 2*std_y:
    result = 'VERY GOOD'
if prediction < mean_y + std_y :
    result = 'GOOD'
if prediction < mean_y:
    result = 'BAD'
if prediction < mean_y - std_y :
    result = 'VERY BAD'
if prediction < mean_y - 2*std_y:
    result = 'TERRIBLE'
if ok:
    st.subheader(f"You have a {result} balance between your work and personal life.")






















