import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Internal imports
from src.data_loader import preprocess_titanic_data
from src.model_trainer import build_and_tune_model

# 1. Load raw data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
test_ids = test_df['PassengerId']

# 2. Process data using our custom loader
train_df = preprocess_titanic_data(train_df)
test_df = preprocess_titanic_data(test_df)

# 3. Prepare for training
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Tune Model (Imports logic from src/model_trainer.py)
best_model = build_and_tune_model(X_train, y_train)

# 5. Evaluate
y_pred = best_model.predict(X_val)
print("\n--- Final Evaluation Report ---")
print(classification_report(y_val, y_pred))

# 6. Generate Submission
final_preds = best_model.predict(test_df)
pd.DataFrame({
    'PassengerId': test_ids, 
    'Survived': final_preds
}).to_csv('submission.csv', index=False)

print("Success! submission.csv created.")