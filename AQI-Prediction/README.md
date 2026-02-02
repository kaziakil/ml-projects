# Air Quality Index (AQI) Prediction using Machine Learning

## 📌 Project Overview
This Project focuses on predicting the **Air Quality Index (AQI)** using machine learning techniques based on historical air pollution data.
The goal is to build a regression model that can estimate AQI values from features such as pollutant concentrations.

---

## 📂 Project Structure 
AQI-Prediction/
┃
┣━ data/
┃	┣━ raw/ # Original Dataset
┃	┗━ processed/ # Cleaned and preprocessed data
┃
┣━ notebooks/
┃	┗━ eda.ipynb # Exploratory Data Analysis
┃
┣━ src/
┃	┣━ preprocess.py # Data cleaning & feature preparation
┃   ┣━ train.py # Model training
┃	┗━ predict.py # AQI prediction
┃
┣━ models
┃	┗━ model.pkl # Trained ML model
┃
┣━ requirements.txt				# Python dependencies
┗━ README.md					# Project Overview, setup, and usage

---

## 📊 Dataset
- Source: Public AQI dataset (e.g., UCI Machine Learning Repository)
- Features may include: 
    - Date
    - Time
    - CO (GT)
    - PT08.S1(CO)
    - NMHC (GT)
    - C6H6 (GT)
    - PT08.S2 (NMHC)
    - NOx(GT)
    - PT08.S3(NOx)
    - NO2(GT)
    - PT08.S4(NO2)
    - PT08.S5(O3)
    - T (Temperature)
    - RH (Relative Humidity)
    - AH (Absolute Humidity)
- Target variable:
    - **AQI**

---

## 🧠 Machine Learning Approach
- Problem Type: **Regression**
- Models Used:
    - Linear Regression
    - Random Forest Regressor
    - XGBoost
    - FB Prophet
- Evaluatoin Metrics:
    - RMSE
    - MAE
    - R² Score


