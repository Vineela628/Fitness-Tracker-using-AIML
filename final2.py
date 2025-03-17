import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# ✅ First Streamlit command - must be at the top!
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏋‍♀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADD BACKGROUND IMAGE ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co/gZJpGGTC/DALL-E-2025-03-17-21-00-26-A-clean-and-minimalistic-sky-blue-background-designed-for-a-fitness-track.webp");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# --- Streamlit UI ---
st.title("🏋‍♂ Personal Fitness Tracker 🧘‍♀")
st.markdown("*Predict your burned calories based on your body parameters!*")

# --- Sidebar Inputs ---
st.sidebar.header("⚙ User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("🎂 Age:", 10, 100, 30)
    bmi = st.sidebar.slider("📊 BMI:", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤ Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡 Body Temperature (C):", 36, 42, 38)
    gender_button = st.sidebar.radio("👥 Gender:", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    return pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender]
    })

df = user_input_features()

st.write("---")
st.header("📌 Your Parameters")
st.write(df)

# --- Loading Animation with Exercise Symbols ---
st.write("🔄 Calculating...")
loading_icons = ["🏋", "🚴", "🧘", "🏃", "🤸", "🏊"]
loading_area = st.empty()
for i in range(10):
    loading_area.markdown(f"## {loading_icons[i % len(loading_icons)]} Processing...")
    time.sleep(0.5)
st.success("✅ Done!")

# --- Load Data & Train Model ---
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge and preprocess data
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

# Calculate BMI
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

# Train/test split
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare data
X_train = pd.get_dummies(exercise_train_data.drop("Calories", axis=1), drop_first=True)
y_train = exercise_train_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
random_reg.fit(X_train, y_train)

# Align user input data with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("🔥 Calories Burned Prediction")
st.success(f"**{round(prediction[0], 2)} kilocalories**")

st.write("---")
st.header("📊 Similar Results")

# Filter similar calorie ranges
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[
    (exercise_df["Calories"] >= calorie_range[0]) & 
    (exercise_df["Calories"] <= calorie_range[1])
]

# Show 5 random examples of similar data
if not similar_data.empty:
    st.write(similar_data.sample(min(5, len(similar_data))))
else:
    st.write("No similar data found!")

st.write("---")
st.header("📢 General Insights")

# Provide comparisons to dataset
st.write(f"You are older than *{round(sum(exercise_df['Age'] < df['Age'].values[0]) / len(exercise_df) * 100, 2)}%* of others.")
st.write(f"Your exercise duration is longer than *{round(sum(exercise_df['Duration'] < df['Duration'].values[0]) / len(exercise_df) * 100, 2)}%* of others.")
st.write(f"Your heart rate is higher than *{round(sum(exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]) / len(exercise_df) * 100, 2)}%* of others.")
st.write(f"Your body temperature is higher than *{round(sum(exercise_df['Body_Temp'] < df['Body_Temp'].values[0]) / len(exercise_df) * 100, 2)}%* of others.")
