# generate_house_data.py

import pandas as pd
import numpy as np

def generate_synthetic_house_data(num_samples=5000, filename='synthetic_house_data.csv'):
    """
    Generates synthetic house price prediction data and saves it to a CSV file.

    The generated data mimics the structure and approximate ranges of the
    'USA Housing' dataset.

    Args:
        num_samples (int): The number of data rows to generate.
        filename (str): The name of the CSV file to save the data to.
    """
    print(f"Generating {num_samples} synthetic house data samples...")

    # Generate features
    # Avg. Area Income: Normally distributed around a mean of 68000
    avg_area_income = np.random.normal(loc=68000, scale=15000, size=num_samples)
    avg_area_income = np.clip(avg_area_income, 20000, 120000) # Clip to realistic range

    # Avg. Area House Age: Normally distributed around a mean of 6 years
    avg_area_house_age = np.random.normal(loc=6, scale=1.5, size=num_samples)
    avg_area_house_age = np.clip(avg_area_house_age, 2, 12) # Clip to realistic range

    # Avg. Area Number of Rooms: Normally distributed around a mean of 6 rooms
    avg_area_num_rooms = np.random.normal(loc=6, scale=1.0, size=num_samples)
    avg_area_num_rooms = np.clip(avg_area_num_rooms, 3, 10) # Clip to realistic range

    # Avg. Area Number of Bedrooms: Normally distributed around a mean of 3 bedrooms
    avg_area_num_bedrooms = np.random.normal(loc=3, scale=0.8, size=num_samples)
    avg_area_num_bedrooms = np.clip(avg_area_num_bedrooms, 1, 5) # Clip to realistic range

    # Area Population: Normally distributed around a mean of 32000
    area_population = np.random.normal(loc=32000, scale=10000, size=num_samples)
    area_population = np.clip(area_population, 5000, 60000) # Clip to realistic range

    # Generate Price (target variable)
    # Price is a linear combination of features plus some noise,
    # mimicking a realistic positive correlation with income, rooms, age, etc.
    # Coefficients are chosen to give reasonable price ranges.
    price = (
        0.8 * avg_area_income +
        15000 * avg_area_house_age +
        120000 * avg_area_num_rooms +
        30000 * avg_area_num_bedrooms +
        8 * area_population +
        np.random.normal(loc=0, scale=50000, size=num_samples) # Add some random noise
    )
    price = np.clip(price, 50000, 2000000) # Ensure prices are somewhat realistic

    # Generate synthetic addresses (random strings)
    addresses = ['{} St, City {}, State {}'.format(
        np.random.randint(100, 999),
        np.random.randint(1000, 9999),
        chr(np.random.randint(65, 90)) + chr(np.random.randint(65, 90))
    ) for _ in range(num_samples)]


    # Create DataFrame
    data = pd.DataFrame({
        'Avg. Area Income': avg_area_income,
        'Avg. Area House Age': avg_area_house_age,
        'Avg. Area Number of Rooms': avg_area_num_rooms,
        'Avg. Area Number of Bedrooms': avg_area_num_bedrooms,
        'Area Population': area_population,
        'Price': price,
        'Address': addresses
    })

    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

if __name__ == "__main__":
    generate_synthetic_house_data(num_samples=5000, filename='synthetic_house_data.csv')
