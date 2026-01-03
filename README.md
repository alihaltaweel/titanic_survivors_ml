# Titanic Survival Prediction

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using structured passenger data. It demonstrates a complete machine learning workflow, including:

* Data cleaning and preprocessing  
* Exploratory Data Analysis (EDA)  
* Feature engineering  
* Model comparison and selection  
* Hyperparameter tuning  
* Evaluation and final prediction generation  

**Dataset:** Kaggle Titanic Dataset

---

## Objectives
* Build a predictive model to classify survival.
* Explore and compare multiple machine learning algorithms.
* Select and fine-tune the best-performing model.
* Evaluate the model on training data and generate test predictions.

---

## Dataset Description
The dataset contains the following columns:

| Column        | Description |
|---------------|-------------|
| PassengerId  | Unique ID for each passenger |
| Survived     | Survival status (0 = No, 1 = Yes) |
| Pclass       | Ticket class (1, 2, 3) |
| Name         | Passenger name |
| Sex          | Gender |
| Age          | Age in years |
| SibSp        | Number of siblings/spouses aboard |
| Parch        | Number of parents/children aboard |
| Ticket       | Ticket number |
| Fare         | Passenger fare |
| Cabin        | Cabin number |
| Embarked     | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Methodology

### 1. Data Cleaning
* Remove duplicates
* Fill missing **Age** with median
* Fill missing **Embarked** with mode
* Fill missing **Fare** with median
* Drop **Cabin** due to too many missing values

---

### 2. Feature Engineering
* Encode **Sex** as numeric (`0 = male`, `1 = female`)
* One-hot encode **Embarked**:
  * `Embarked_C`
  * `Embarked_Q`
  * `Embarked_S`

---

### 3. Model Selection
Models tested:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)
* Gradient Boosting

* Used **5-fold cross-validation**
* **Best model:** Gradient Boosting (~83% CV accuracy)

---

### 4. Hyperparameter Tuning
Grid search over:

* `n_estimators`: 
* `learning_rate`: 
* `max_depth`: 

**Best parameters:**
* `n_estimators = `
* `learning_rate = `
* `max_depth = `

---

### 5. Evaluation Metrics
* **Accuracy:** 
* **Precision:**
* **Recall:** 
* **F1-score:** 

**Confusion Matrix:**
---

## Results
* Final predictions are saved as:
* Gradient Boosting provides the best balance between precision and recall.

---

## How to Run

### 1. Prepare the Data
Place `train.csv` and `test.csv` in the `data/` folder.

---

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate