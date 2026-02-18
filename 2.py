import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # Changed here
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'cropdata_updated.csv'  # Update with your file path
crop_data = pd.read_csv(file_path)

# Define feature and target columns
features = crop_data.drop(columns=['result'])
target = crop_data['result']

# Identify categorical and numerical columns
categorical_columns = ['crop ID', 'soil_type', 'Seedling Stage']
numerical_columns = ['MOI', 'temp', 'humidity']

# Set up preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Preprocess features
X = preprocessor.fit_transform(features)
y = target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier (changed to RandomForestClassifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy:.2%}")

# Generate random data for prediction
random_data = {
    'crop ID': ['Wheat'],
    'soil_type': ['Black Soil'],
    'Seedling Stage': ['Germination'],
    'MOI': [np.random.randint(1, 10)],
    'temp': [np.random.randint(20, 40)],
    'humidity': [np.random.uniform(50, 100)]
}

# Convert to DataFrame and ensure order
random_df = pd.DataFrame(random_data)
random_df = random_df[features.columns]

# Preprocess and predict
random_data_preprocessed = preprocessor.transform(random_df)
predicted_result = classifier.predict(random_data_preprocessed)

print(f"Random Data: {random_data}")
print(f"Predicted Result: {'Success' if predicted_result[0] == 1 else 'Failure'}")
