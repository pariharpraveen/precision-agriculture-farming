import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Optional: XGBoost (install with `pip install xgboost`)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Load the dataset
file_path = 'cropdata_updated.csv'  # Make sure this path is correct
crop_data = pd.read_csv(file_path)

# Define features and target
features = crop_data.drop(columns=['result'])
target = crop_data['result']

# Define column types
categorical_columns = ['crop ID', 'soil_type', 'Seedling Stage']
numerical_columns = ['MOI', 'temp', 'humidity']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Apply preprocessing
X = preprocessor.fit_transform(features)
y = target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of classifiers to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(eval_metric='logloss', random_state=42)

# Train and evaluate all models
print("Model Accuracy Comparison:\n---------------------------")
for name, model in models.items():
    # GaussianNB requires dense input
    if isinstance(model, GaussianNB):
        model.fit(X_train.toarray(), y_train)
        y_pred = model.predict(X_test.toarray())
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2%}")

# Use the best performing model (you can manually pick one)
chosen_model = models["Random Forest"]  # Change to your preferred model

# Predict on random sample
random_data = {
    'crop ID': ['Wheat'],
    'soil_type': ['Black Soil'],
    'Seedling Stage': ['Germination'],
    'MOI': [np.random.randint(1, 10)],
    'temp': [np.random.randint(20, 40)],
    'humidity': [np.random.uniform(50, 100)]
}

random_df = pd.DataFrame(random_data)
random_df = random_df[features.columns]
random_data_preprocessed = preprocessor.transform(random_df)

# Convert if needed for GaussianNB
if isinstance(chosen_model, GaussianNB):
    random_data_preprocessed = random_data_preprocessed.toarray()

predicted_result = chosen_model.predict(random_data_preprocessed)

print("\nRandom Data Prediction:")
print(random_data)
print(f"Predicted Result: {'Success' if predicted_result[0] == 1 else 'Failure'}")
