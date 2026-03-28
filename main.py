import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error # type: ignore
import pickle

# Load the dataset
df = pd.read_csv('house_prediction.csv')

# Features: hall, bedroom, kitchen, sqft, floor, bathroom, garden, parking, pooja_room
# Target: price

print("Dataset Preview:")
print(df.head())

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Initialize and train regressors
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Evaluate the models
for name, model in models.items():
    model.fit(X, y)
    y_pred = np.round(model.predict(X))
    print(f"\n{name}:")
    print("R2 Score (Accuracy):", r2_score(y, y_pred) * 100, "%")
    print("Mean Absolute Error:", mean_absolute_error(y, y_pred))

# Save the best model for hosting
best_model = models['Decision Tree']  # Decision Tree works best for exact value lookup in small datasets
with open('house_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nModel trained and saved as house_model.pkl")