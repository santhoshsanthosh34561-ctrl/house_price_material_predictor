import pandas as pd
import os

# Load the existing dataset
df = pd.read_csv('house_prediction.csv')

# Base Rate is 2300 per sqft
base_rate = 2300

# Update the price column using the user's NEW formula
# Price = (Sqft × Rate) + (Bedroom × 30000) + (Bathroom × 20000) + (Parking × 30000) + (Garden × 80000) + (Floor × 150000)
df['price'] = (df['sqft'] * base_rate) + \
              (df['bedroom'] * 30000) + \
              (df['bathroom'] * 20000) + \
              (df['parking'] * 30000) + \
              (df['garden'] * 80000) + \
              ((df['floor'] - 1) * 150000)

# Note: Hall, Kitchen, and Pooja Room are no longer in the price formula as per latest request.

# Save the updated dataset
df.to_csv('house_prediction.csv', index=False)
print("Dataset updated successfully with latest pricing model!")

# Remove the cached model so it retrains with the new data
# Renaming to v4 to ensure cache clear
if os.path.exists("house_model_v3.pkl"):
    os.remove("house_model_v3.pkl")
print("Deleted old model so it can be retrained.")
