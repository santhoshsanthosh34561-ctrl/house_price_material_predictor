import pandas as pd
import os

# Load the existing dataset
df = pd.read_csv('house_prediction.csv')

# Base Rate is 2300 per sqft
base_rate = 2300

# Update the price column using the user's NEW formula
# Price = (sqft * 2300) + (bedroom * 60000) + (bathroom * 30000) + (kitchen * 50000) + (floor * 120000) + (parking * 80000) + (garden_area * 100) + (pooja_room * 30000)
df['price'] = (df['sqft'] * base_rate) + \
              (df['bedroom'] * 60000) + \
              (df['bathroom'] * 30000) + \
              (df['kitchen'] * 50000) + \
              (df['floor'] * 120000) + \
              (df['parking'] * 80000) + \
              (df['garden_area'] * 100) + \
              (df['pooja_room'] * 30000)

# Save the updated dataset
df.to_csv('house_prediction.csv', index=False)
print("Dataset updated successfully with latest pricing model!")

# Remove old models to ensure retrain
for v in range(1, 7):
    if os.path.exists(f"house_model_v{v}.pkl"):
        os.remove(f"house_model_v{v}.pkl")
print("Deleted old models so they can be retrained.")
