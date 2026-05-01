import pandas as pd
import os

# Load the existing dataset
df = pd.read_csv('house_prediction.csv')

# Base Rate is 2300 per sqft
base_rate = 2300

# Update the price column using the user's formula
# The user wants: Floor_Cost = Sqft × 2200 for every additional floor
df['price'] = (df['sqft'] * base_rate) + \
              (df['bedroom'] * 50000) + \
              (df['bathroom'] * 30000) + \
              (df['hall'] * 40000) + \
              (df['kitchen'] * 30000) + \
              ((df['floor'] - 1) * df['sqft'] * 2200) + \
              (df['parking'] * 60000) + \
              (df['garden'] * 120000) + \
              (df['pooja_room'] * 40000)

# Save the updated dataset
df.to_csv('house_prediction.csv', index=False)
print("Dataset updated successfully with new floor logic!")

# Remove the cached model so it retrains with the new data
if os.path.exists("house_model_v2.pkl"):
    os.remove("house_model_v2.pkl")
if os.path.exists("house_model_v3.pkl"):
    os.remove("house_model_v3.pkl")
print("Deleted old model so it can be retrained.")
