import pandas as pd
import os

# Load the existing dataset
df = pd.read_csv('house_prediction.csv')

# Base Rate is 2300 per sqft
base_rate = 2300

# Update the price column using the user's NEW fixed formula
# Price = (Sqft × Rate) + (Hall × 80000) + (Bedroom × 120000) + (Kitchen × 100000) + (Bathroom × 60000) + (Floor × 200000) + (Parking × 100000) + (Garden × 150000) + (PoojaRoom × 50000)
df['price'] = (df['sqft'] * base_rate) + \
              (df['hall'] * 80000) + \
              (df['bedroom'] * 120000) + \
              (df['kitchen'] * 100000) + \
              (df['bathroom'] * 60000) + \
              (df['floor'] * 200000) + \
              (df['parking'] * 100000) + \
              (df['garden'] * 150000) + \
              (df['pooja_room'] * 50000)

# Save the updated dataset
df.to_csv('house_prediction.csv', index=False)
print("Dataset updated successfully with latest pricing model!")

# Remove old models to ensure retrain
for v in range(1, 6):
    if os.path.exists(f"house_model_v{v}.pkl"):
        os.remove(f"house_model_v{v}.pkl")
print("Deleted old models so they can be retrained.")
