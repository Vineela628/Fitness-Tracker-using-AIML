import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
st.write("In this WebApp, you can predict the calories burned based on your exercise parameters such as `Age`, `Gender`, `BMI`, etc.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    exercise_type = st.sidebar.selectbox("Exercise Type", ["Running", "Cycling", "Walking", "Swimming"])
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    
    gender = 1 if gender_button == "Male" else 0
    
    data_model = {
        "Exercise_Type": exercise_type,
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column
def calculate_bmi(data):
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    return data

exercise_train_data = calculate_bmi(exercise_train_data)
exercise_test_data = calculate_bmi(exercise_test_data)

# Prepare the data
selected_columns = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
exercise_train_data = exercise_train_data[selected_columns]
exercise_test_data = exercise_test_data[selected_columns]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results: ")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")
st.write("You are older than", round(sum((exercise_df["Age"] < df["Age"].values[0])) / len(exercise_df) * 100, 2), "% of other people.")
st.write("Your exercise duration is higher than", round(sum((exercise_df["Duration"] < df["Duration"].values[0])) / len(exercise_df) * 100, 2), "% of other people.")
st.write("You have a higher heart rate than", round(sum((exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0])) / len(exercise_df) * 100, 2), "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum((exercise_df["Body_Temp"] < df["Body_Temp"].values[0])) / len(exercise_df) * 100, 2), "% of other people during exercise.")
