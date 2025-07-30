# House Price Prediction

This repository contains code for predicting house prices using various Machine Learning models.

## Table of Contents

-   [Overview](#overview)
-   [Dataset](#dataset)
-   [Requirements](#requirements)
-   [Project Structure](#project-structure)
-   [How to Run](#how-to-run)
    -   [Jupyter Notebook](#jupyter-notebook)
    -   [Python Script](#python-script)
-   [Models Used](#models-used)
-   [Results](#results)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

This project aims to predict house prices based on several features such as average area income, house age, number of rooms, number of bedrooms, and area population. The process involves:
1.  **Exploratory Data Analysis (EDA)**: Understanding the dataset's structure, distributions, and relationships.
2.  **Data Preprocessing**: Handling missing values, scaling numerical features.
3.  **Model Training**: Training various regression models (e.g., Linear Regression, Random Forest, Gradient Boosting).
4.  **Model Evaluation**: Assessing model performance using metrics like MAE, MSE, RMSE, and R-squared.
5.  **Prediction**: Using the trained model to predict prices for new, unseen data.

## Dataset

The dataset used for this project is the **USA Housing** dataset. It contains the following columns:

-   `Avg. Area Income`: Average income of residents of the city where the house is located.
-   `Avg. Area House Age`: Average age of houses in the same city.
-   `Avg. Area Number of Rooms`: Average number of rooms in houses in the same city.
-   `Avg. Area Number of Bedrooms`: Average number of bedrooms in houses in the same city.
-   `Area Population`: Population of the city.
-   `Price`: Price of the house (Target variable).
-   `Address`: Address of the house (dropped in preprocessing).

**Download Link:**
You can download the dataset from Kaggle: [USA Housing Dataset](https://www.kaggle.com/datasets/vedavyasv/usa-housing)

Please download the `USA_Housing.csv` file and place it in the same directory as the Python scripts/notebook.

## Requirements

To run this project, you need Python 3.x and the following libraries:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `joblib`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
## Project Structure
```bash
├── housepriceprediction.ipynb  # Jupyter Notebook for full workflow (EDA, Training, Evaluation)
├── houseprediction.py          # Python script for making predictions using a saved model
├── USA_Housing.csv             # The dataset file (download separately)
├── house_price_prediction_model.pkl # Saved trained model (generated after running the notebook)
└── README.md                   # This README file
```
## How to Run
I can certainly help you with the code for house price prediction using Python and Machine Learning, along with a README.md and a dataset link.

Please note that directly providing .ipynb or .py files is not possible in this text-based format. However, I will give you the complete code for both, which you can then save as housepriceprediction.ipynb (if you're using Jupyter Notebook) and houseprediction.py.

Here's the breakdown:

housepriceprediction.ipynb (Jupyter Notebook Code): This will include all the steps: data loading, exploration, preprocessing, model training, and evaluation.

houseprediction.py (Script Version): A simplified version focusing on the core prediction logic.

README.md: Explains the project, how to run it, and the dataset.

Dataset Link: A commonly used dataset for house price prediction.

1. housepriceprediction.ipynb (Jupyter Notebook Code)
This code will use a common machine learning workflow. I'll use scikit-learn for simplicity and pandas for data manipulation.

Python

# housepriceprediction.ipynb

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 2. Load Dataset
# You can download the dataset from the link provided in README.md
# For this example, let's assume 'USA_Housing.csv' is in the same directory.
try:
    df = pd.read_csv('USA_Housing.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: USA_Housing.csv not found. Please ensure the dataset is in the same directory.")
    print("Download it from: https://www.kaggle.com/datasets/vedavyasv/usa-housing")
    exit() # Exit if dataset is not found

# 3. Exploratory Data Analysis (EDA)
print("\n--- EDA ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum()) # Check for missing values

print("\nDescriptive statistics:")
print(df.describe())

# Drop the 'Address' column as it's likely not useful for prediction
if 'Address' in df.columns:
    df = df.drop('Address', axis=1)
    print("\n'Address' column dropped.")

# Visualize distributions (example for numerical columns)
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.select_dtypes(include=np.number).columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 4. Data Preprocessing
print("\n--- Data Preprocessing ---")

# Define features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Identify numerical and categorical features (all are numerical in this dataset)
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist() # Should be empty for USA_Housing

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler() # Standardize numerical features

# For USA_Housing, there are no categorical features, but including this for generalizability
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encode categorical features

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns not specified
)

# 5. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 6. Model Training
print("\n--- Model Training ---")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")

    trained_models[name] = pipeline

# Save the best performing model (e.g., Random Forest)
best_model_name = 'Random Forest Regressor' # Or choose based on R2 score
if best_model_name in trained_models:
    joblib.dump(trained_models[best_model_name], 'house_price_prediction_model.pkl')
    print(f"\n{best_model_name} model saved as house_price_prediction_model.pkl")
else:
    print(f"\nWarning: '{best_model_name}' not found in trained models. Model not saved.")


# 7. Make Predictions (Example with a new data point)
print("\n--- Making Predictions ---")

# Create a sample new data point (ensure it has the same columns as X)
# Example values, adjust as needed
sample_data = pd.DataFrame([[80000, 7, 6, 250000, 4.5]], # Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population
                           columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                                    'Avg. Area Number of Bedrooms', 'Area Population'])

print("\nSample data for prediction:")
print(sample_data)

# Load the saved model
try:
    loaded_model = joblib.load('house_price_prediction_model.pkl')
    predicted_price = loaded_model.predict(sample_data)
    print(f"\nPredicted price for the sample data: ${predicted_price[0]:,.2f}")
except FileNotFoundError:
    print("\nError: Model file 'house_price_prediction_model.pkl' not found. Please train and save the model first.")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")

# Optional: Visualize Residuals for the best model
if best_model_name in trained_models:
    best_model_pipeline = trained_models[best_model_name]
    y_pred_best = best_model_pipeline.predict(X_test)
    residuals = y_test - y_pred_best

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution for {best_model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs. Predicted Values for {best_model_name}')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.show()

print("\nHouse Price Prediction Notebook execution complete.")
To use this in Jupyter Notebook:

Save the above code as a new Jupyter Notebook file (e.g., housepriceprediction.ipynb).

Open Jupyter Notebook, navigate to the directory where you saved the file, and open it.

Run the cells sequentially.

2. houseprediction.py (Script Version)
This script will focus on loading a pre-trained model and making predictions.

Python

# houseprediction.py

import pandas as pd
import joblib
import sys
import os

def predict_house_price(income, house_age, num_rooms, num_bedrooms, population):
    """
    Predicts house price using a pre-trained model.

    Args:
        income (float): Average Area Income
        house_age (float): Average Area House Age
        num_rooms (float): Average Area Number of Rooms
        num_bedrooms (float): Average Area Number of Bedrooms
        population (float): Area Population

    Returns:
        float: Predicted house price, or None if an error occurs.
    """
    model_path = 'house_price_prediction_model.pkl'

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        print("Please run 'housepriceprediction.ipynb' first to train and save the model.", file=sys.stderr)
        return None

    try:
        loaded_model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

    # Create a DataFrame for the input features, ensuring column order matches training data
    # The column names MUST match those used during training
    input_data = pd.DataFrame([[income, house_age, num_rooms, num_bedrooms, population]],
                              columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                                       'Avg. Area Number of Bedrooms', 'Area Population'])

    try:
        predicted_price = loaded_model.predict(input_data)[0]
        return predicted_price
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    # Example usage:
    # Ensure you've run housepriceprediction.ipynb first to save the model.
    print("--- House Price Prediction Script ---")

    # Get input from user (optional, can be hardcoded for testing)
    try:
        avg_area_income = float(input("Enter Average Area Income: "))
        avg_area_house_age = float(input("Enter Average Area House Age: "))
        avg_area_num_rooms = float(input("Enter Average Area Number of Rooms: "))
        avg_area_num_bedrooms = float(input("Enter Average Area Number of Bedrooms: "))
        area_population = float(input("Enter Area Population: "))

        predicted_price = predict_house_price(avg_area_income, avg_area_house_age,
                                              avg_area_num_rooms, avg_area_num_bedrooms,
                                              area_population)

        if predicted_price is not None:
            print(f"\nPredicted House Price: ${predicted_price:,.2f}")
        else:
            print("\nCould not predict house price. Please check the error messages above.")

    except ValueError:
        print("Invalid input. Please enter numeric values.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

To use this script:

Save the above code as houseprediction.py.

Make sure you have run the housepriceprediction.ipynb notebook at least once to train and save the house_price_prediction_model.pkl file in the same directory.

Run from your terminal: python houseprediction.py
