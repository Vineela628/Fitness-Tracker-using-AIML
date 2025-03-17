.

# 🏋️ Personal Fitness Tracker 🏃‍♀
📋 Project Overview
The Personal Fitness Tracker is a machine learning-powered web application that predicts the number of calories burned during physical exercise. Users input personal details such as age, BMI, exercise duration, and heart rate, and the app provides an accurate calorie-burning estimate. Built with Streamlit, the application offers an intuitive, interactive interface with helpful insights based on a real dataset.

## ✨ Features
✅ Predict calories burned based on user input
✅ Analyze user parameters in comparison to others
✅ View similar records from the dataset
✅ Interactive sliders and radio buttons for inputs
✅ Login system for user authentication (in one version)
✅ Clean UI with customizable themes
✅ Fast predictions with a Random Forest model
✅ Real-time progress indicators and loading animations

## 📂 Folder Structure
bash
Copy
Edit
├── app.py              # Streamlit web app (main version)
├── dfit.py             # Alternate version of the web app
├── final2.py           # Enhanced UI version of the web app
├── calories.csv        # Dataset with calorie information
├── exercise.csv        # Dataset with exercise details
├── requirements.txt    # Python dependencies
## 🏗️ Tech Stack
Frontend & UI: Streamlit

Backend & ML: Python, Pandas, Numpy, Scikit-learn

Visualization: Seaborn, Matplotlib

Model: Random Forest Regressor

## 🔍 How It Works
User Input Parameters:
Users provide inputs such as Age, Gender, BMI, Exercise Duration, Heart Rate, and Body Temperature.

Data Preprocessing:
The app merges and preprocesses two datasets (calories.csv and exercise.csv). BMI is calculated, and features are prepared for model training.

Model Training:
A Random Forest Regression model is trained to predict calorie expenditure using selected features.

Prediction & Insights:
The model predicts calories burned. Users can view similar cases and get comparative insights about their performance.

## 🚀 Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.10+

pip (Python package manager)

## Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/Vineela628/personal-fitness-tracker-using-AIML
cd personal-fitness-tracker
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App
You can run any of the app files, for example:

bash
Copy
Edit
streamlit run app.py
## 🧪 Dataset Details
calories.csv
User_ID

Calories (target variable)

exercise.csv
User_ID

Gender

Age

Height

Weight

Duration

Heart_Rate

Body_Temp

Both datasets are merged on User_ID for building the training data.

## ⚙️ Model Information
Model: Random Forest Regressor

Hyperparameters:

n_estimators=1000

max_features=3

max_depth=6

Training/Test Split: 80/20 split

Feature Engineering:

BMI calculated from height and weight

One-hot encoding for gender (male/female)

## 💻 Usage
Open the web app in your browser after running the script.

Input your personal exercise data via the sidebar widgets:

Age

BMI

Exercise Duration

Heart Rate

Body Temperature

Gender

View predicted calories burned.

Explore comparative insights and similar records from the dataset.

 Login functionality is provided in the app.py version.


## 📝 Requirements
Listed in requirements.txt:

shell
Copy
Edit
streamlit>=1.42.2
numpy>=2.2.3
pandas>=2.2.3
scikit-learn>=1.6.1
Install them via:

bash
Copy
Edit
pip install -r requirements.txt
### 🙌 Credits
Developed by Vineela.c

Dataset sourced from [Kaggle / Custom Dataset]

Special thanks to the Streamlit and Scikit-learn communities









# Fitness-Tracker-using-AIML
This Personal Fitness Tracker web app predicts calories burned based on user inputs like age, BMI, heart rate, and exercise duration, using machine learning models and interactive Streamlit features.
