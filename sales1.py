import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the dataset (replace with the correct path to your dataset)
data = pd.read_csv("advertising.csv")  # Ensure the path to your CSV file is correct
# Explore the dataset
print(data.head())
print(data.info())
# Step 1: Data Preprocessing (Handle Missing Values)
data.fillna(0, inplace=True)
# Step 2: Exploratory Data Analysis (EDA)
# Visualize the correlation between features and target
sns.pairplot(data)
plt.show()
# Correlation matrix to understand the relationships between variables
corr_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
# Step 3: Feature Selection and Data Split
X = data[['TV', 'Radio', 'Newspaper']]  # Features (ad spend)
y = data['Sales']  # Target variable (sales)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Model Selection and Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Step 5: Model Evaluation
y_pred = model.predict(X_test)
# Evaluate the performance of the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
# Step 6: Get user input for future sales prediction
def get_user_input():
    print("Enter the advertising spend for TV, Radio, and Newspaper (in units of your currency):")
    try:
        tv_spend = float(input("TV Advertising Spend: "))
        radio_spend = float(input("Radio Advertising Spend: "))
        newspaper_spend = float(input("Newspaper Advertising Spend: "))
        
        # Prepare the input for prediction
        user_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
        return user_data
    except ValueError:
        print("Invalid input. Please enter numerical values for advertising spend.")
        return None
# Step 7: Predicting future sales based on user input
user_input = get_user_input()
if user_input is not None:
    sales_prediction = model.predict(user_input)
    print(f"Predicted Sales: {sales_prediction[0]:.2f}")
