import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from src.data_loader import preprocess_titanic_data

# 1. Load Data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
test_passenger_ids = test_df['PassengerId'] # Save for final submission

# 2. Apply Custom Engineering
train_df = preprocess_titanic_data(train_df)
test_df = preprocess_titanic_data(test_df)

# 3. Split Features and Target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define Pipeline
numeric_features = ['Age', 'Fare', 'Family_Size']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Build and Train Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_val)
print("--- Validation Metrics ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))

# 7. Final Predictions for Kaggle/Submission
predictions = model.predict(test_df)
output = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")