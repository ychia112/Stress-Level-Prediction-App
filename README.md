# Stress Level & Work-Life Balance Prediction App

This is a Python-based web application built with **Streamlit** that predicts an individual's **stress level** and **work-life balance indicator** based on user responses to a short questionnaire.

The project leverages **supervised learning models** including **Linear Regression** for stress level prediction and **classification models** with Adaboost classifier for work-life balance categorization. It also includes key **feature engineering** steps such as encoding categorical variables to ensure accurate model performance.

## Motivation

We found that the daily stress level is not easy to evaluate, so we build a classification model to predict the possible
stress level of the user, and use the prediction to calculate the Work Life Balance score.

## Features

- **Stress Level Prediction** using a Linear Regression model
- **Work-Life Balance Classification** using a classification model
- **Interactive Questionnaire Interface** built with Streamlit

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib

## Source

The dataset is from kaggle:"kaggle datasets download -d ydalat/lifestyle-and-wellbeing-data"
https://www.kaggle.com/datasets/ydalat/lifestyle-and-wellbeing-data

Try it with your own data and see the prediction result!
https://ychia112-aiproject-regression-app-byp76o.streamlit.app/
