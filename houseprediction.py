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
