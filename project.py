import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
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

# Set up preprocessing for categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Preprocess the data
X = preprocessor.fit_transform(features)
y = target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Decision Tree Classifier: {accuracy:.2%}")

# Generate random data for prediction
random_data = {
    'crop ID': ['Wheat'],  # You can change this to another crop
    'soil_type': ['Black Soil'],  # Change this to a different soil type if needed
    'Seedling Stage': ['Germination'],  # Select a valid growth stage
    'MOI': [np.random.randint(1, 10)],  # Random MOI between 1 and 10
    'temp': [np.random.randint(20, 40)],  # Random temperature between 20 and 40
    'humidity': [np.random.uniform(50, 100)]  # Random humidity between 50% and 100%
}

# Convert random data to DataFrame
random_df = pd.DataFrame(random_data)

# Ensure random_df columns match the training features
random_df = random_df[features.columns]

# Preprocess the random data
random_data_preprocessed = preprocessor.transform(random_df)

# Predict the result
predicted_result = classifier.predict(random_data_preprocessed)
print(f"Random Data: {random_data}")
print(f"Predicted Result: {'Success' if predicted_result[0] == 1 else 'Failure'}")
