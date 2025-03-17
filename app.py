import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

# Custom CSS for dark red theme
st.markdown(
    """
    <style>
        body {
            background-color: #3B0000;
            color: white;
        }
        .stApp {
            background-color: #3B0000;
        }
        .sidebar .sidebar-content {
            background-color: #5E0000;
        }
        .stButton>button {
            background-color: #5E0000;
            color: white;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            background-color: #5E0000;
            color: white;
        }
        .stProgress>div>div>div {
            background-color: #5E0000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Login System
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    email = st.text_input("Email:", key="email")
    password = st.text_input("Password:", type="password", key="password")
    if st.button("Login"):
        if email and password:  # Simple validation
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.warning("Please enter both email and password")

if not st.session_state.authenticated:
    st.title("Login Page")
    login()
else:
    st.title("Personal Fitness Tracker")
    st.write("Input parameters to predict calories burned.")
    st.sidebar.header("User Input Parameters")
    
    def user_input_features():
        age = st.sidebar.slider("Age:", 10, 100, 30)
        bmi = st.sidebar.slider("BMI:", 15, 40, 20)
        duration = st.sidebar.slider("Duration (min):", 0, 35, 15)
        heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
        body_temp = st.sidebar.slider("Body Temperature (C):", 36, 42, 38)
        gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))

        gender = 1 if gender_button == "Male" else 0

        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": gender
        }

        return pd.DataFrame(data_model, index=[0])

    df = user_input_features()
    st.write("---")
    st.header("Your Parameters:")
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df)

    # Load and preprocess data
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)

    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)
    
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    
    df = df.reindex(columns=X_train.columns, fill_value=0)
    prediction = random_reg.predict(df)
    
    st.write("---")
    st.header("Prediction:")
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    
    st.write(f"{round(prediction[0], 2)} *kilocalories*")
    
    st.write("---")
    st.header("Similar Results:")
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))
    
    st.write("---")
    st.header("General Information:")
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
    
    st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
    st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
    st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
    st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")