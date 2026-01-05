# 🚢 Titanic Survival Prediction

A comprehensive machine learning project that predicts passenger survival on the Titanic based on various demographic and socio-economic features.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Project Overview

This project uses supervised machine learning techniques to predict whether a passenger survived the Titanic disaster. The model analyzes historical passenger data including demographic information, ticket class, family relationships, and fare prices to make predictions.

**Goal**: Build an accurate classification model that can predict survival outcomes with high precision and recall.

**Achievement**: The final tuned Logistic Regression model achieved **86.03% accuracy** on the validation set with an F1 score of 0.81.

## 📦 Dataset

The dataset is from the famous [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle.

### Files:
- `train.csv`: Training data with survival labels (891 passengers)
- `test.csv`: Test data without labels (418 passengers)

### Features:

| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | Integer | Unique ID for each passenger |
| Pclass | Integer | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | String | Passenger name |
| Sex | String | Gender (male/female) |
| Age | Float | Age in years |
| SibSp | Integer | Number of siblings/spouses aboard |
| Parch | Integer | Number of parents/children aboard |
| Ticket | String | Ticket number |
| Fare | Float | Passenger fare |
| Cabin | String | Cabin number |
| Embarked | String | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

### Target Variable:
- `Survived`: 0 = Did not survive, 1 = Survived

## 📁 Project Structure

```
titanic-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── output/
│   └── submission.csv
│
├── titanic_survival_prediction.py
├── README.md
└── requirements.txt
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-prediction.git
cd titanic-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## 🚀 Usage

1. Ensure the dataset files (`train.csv` and `test.csv`) are in the `data/` directory
2. Run the complete workflow script:
```bash
python titanic_survival_prediction.py
```

3. The script will:
   - Perform exploratory data analysis with visualizations
   - Preprocess and engineer features
   - Train and compare 6 different models
   - Tune hyperparameters for the best model
   - Generate predictions and create `submission.csv` in the `output/` folder

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

**Visualizations Created**:
- Correlation heatmap for numeric features
- Overall survival count distribution
- Survival rates by gender
- Survival rates by passenger class
- Age distribution histogram
- Fare distribution histogram
- Survival rates by port of embarkation

**Key Findings**:
- Women had significantly higher survival rates than men
- First-class passengers survived at higher rates
- Age and family size influenced survival chances

### 2. Data Preprocessing & Feature Engineering

**Handling Missing Values**:
- `Age`: Filled using median age grouped by Title and Pclass
- `Embarked`: Filled with mode (most frequent value)
- `Fare`: Filled with median fare
- `Cabin`: Dropped due to 77% missing values

**Feature Engineering**:
- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, Master, Rare) from passenger names
- **Family Size**: Combined SibSp and Parch (`FamilySize = SibSp + Parch + 1`)
- **IsAlone**: Binary feature indicating solo travelers (1 if alone, 0 if with family)
- **Age Bins**: Categorized ages into 5 groups (Child, Teen, Adult, Middle, Senior)
- **Fare Bins**: Quartile-based fare categories (Low, Medium, High, Very High)

**Encoding & Scaling**:
- One-hot encoding for categorical variables (Sex, Embarked, Title, AgeBin, FareBin)
- StandardScaler applied to all features for model consistency
- Maintained alignment between train and test sets using combined preprocessing

### 3. Model Building & Comparison

Trained and evaluated **6 classification algorithms**:

1. **Logistic Regression**: Simple, interpretable baseline
2. **Random Forest**: Ensemble of decision trees
3. **Gradient Boosting**: Sequential boosting technique
4. **Support Vector Machine (SVM)**: Kernel-based classifier
5. **K-Nearest Neighbors (KNN)**: Instance-based learning
6. **XGBoost**: Advanced gradient boosting framework

**Training Strategy**:
- 80/20 train-validation split with stratification
- 5-fold cross-validation for robust performance estimation
- StandardScaler for feature normalization

### 4. Model Evaluation

**Metrics Used**:
- **Accuracy**: Overall correctness of predictions
- **Cross-Validation Score**: Average accuracy across 5 folds
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of true/false positives/negatives

### 5. Hyperparameter Tuning

Applied **GridSearchCV** to optimize the best-performing model:
- Tested multiple hyperparameter combinations
- Used 5-fold cross-validation during grid search
- Selected optimal parameters based on cross-validated performance

## 📊 Results

### Model Performance Comparison

| Model | Validation Accuracy | CV Score | F1 Score |
|-------|---------------------|----------|----------|
| **Logistic Regression** | **86.03%** | **81.61%** | **0.812** |
| Gradient Boosting | 83.24% | 81.19% | 0.766 |
| SVM | 82.68% | 82.30% | 0.756 |
| Random Forest | 82.12% | 79.79% | 0.761 |
| XGBoost | 81.56% | 78.80% | 0.759 |
| KNN | 75.98% | 79.64% | 0.677 |

### Best Model: Logistic Regression

**Optimal Hyperparameters** (after GridSearchCV):
- `C = 0.1` (regularization strength)

**Performance Metrics**:
- **Validation Accuracy**: 86.03%
- **Cross-Validation Score**: 81.61%
- **F1 Score**: 0.812

**Why Logistic Regression Won**:
- Best balance between bias and variance
- Strong regularization (C=0.1) prevented overfitting
- Worked well with engineered features and scaled data
- Simple, interpretable model with excellent generalization

### Feature Importance

Based on Logistic Regression coefficients, the most influential features were:
1. **Sex (male)**: Strong negative coefficient (being male decreased survival probability)
2. **Pclass**: Higher class significantly increased survival chances
3. **Title**: Titles like "Mr" vs "Mrs/Miss" captured gender and social status
4. **Age**: Younger passengers had better survival rates
5. **Fare**: Higher fares correlated with better survival (proxy for class/wealth)

## 💡 Key Insights

1. **Gender Disparity**: Women had ~75% survival rate vs ~19% for men, reflecting the "Women and children first" evacuation policy

2. **Class-Based Survival**: First-class passengers had ~63% survival rate compared to ~24% for third-class passengers

3. **Age Factor**: Children and younger passengers had higher survival rates across all classes

4. **Family Dynamics**: Passengers with small families (2-4 members) survived at higher rates than solo travelers or very large families

5. **Port of Embarkation**: Passengers boarding at Cherbourg had slightly higher survival rates, likely correlated with higher-class tickets

6. **Title Engineering**: Extracting titles from names proved highly predictive, capturing both gender and social status information

## 🔮 Future Improvements

1. **Advanced Feature Engineering**:
   - Extract deck information from cabin numbers for passengers with cabin data
   - Create interaction features (e.g., Sex × Pclass, Age × FamilySize)
   - Engineer features based on ticket numbers or cabin locations

2. **Model Enhancements**:
   - Implement ensemble methods (stacking, blending multiple models)
   - Try deep learning with neural networks
   - Experiment with CatBoost or LightGBM

3. **Data Augmentation**:
   - Use external historical data about the Titanic disaster
   - Apply SMOTE for handling class imbalance
   - Implement more sophisticated imputation methods (MICE, KNN imputation)

4. **Error Analysis**:
   - Deep dive into misclassified passengers
   - Identify patterns in false positives and false negatives
   - Analyze prediction confidence scores

5. **Model Interpretability**:
   - Generate SHAP values for better feature interpretation
   - Create partial dependence plots
   - Build LIME explanations for individual predictions

## 📚 Learning Outcomes

This project successfully demonstrates:
- ✅ Complete end-to-end machine learning workflow
- ✅ Comprehensive EDA with meaningful visualizations
- ✅ Advanced data preprocessing and feature engineering
- ✅ Comparison of 6 different classification algorithms
- ✅ Cross-validation for robust model evaluation
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Understanding of classification metrics (accuracy, F1, confusion matrix)
- ✅ Insights into historical bias and its impact on predictions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic)
- Inspired by the historical Titanic disaster and the data science community
- Built as a supervised learning project for machine learning certification

---

**Project Status**: ✅ Complete | **Kaggle Score**: Ready for submission

**Happy Learning! 🚀**