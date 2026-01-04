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
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project uses supervised machine learning techniques to predict whether a passenger survived the Titanic disaster. The model analyzes historical passenger data including demographic information, ticket class, family relationships, and fare prices to make predictions.

**Goal**: Build an accurate classification model that can predict survival outcomes with high precision and recall.

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
├── notebooks/
│   └── titanic_analysis.ipynb
│
├── outputs/
│   ├── submission.csv
│   ├── eda_visualizations.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
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
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 🚀 Usage

1. Ensure the dataset files (`train.csv` and `test.csv`) are in the `data/` directory
2. Open `titanic_analysis.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. The notebook will:
   - Perform exploratory data analysis
   - Preprocess and engineer features
   - Train multiple models
   - Evaluate performance
   - Generate predictions
   - Create `submission.csv` for Kaggle submission

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed survival rates across different demographics
- Identified class imbalance and missing values
- Visualized relationships between features and survival

**Key Findings**:
- Women had significantly higher survival rates than men
- First-class passengers survived at higher rates
- Age and family size influenced survival chances

### 2. Data Preprocessing

**Handling Missing Values**:
- `Age`: Filled using median age grouped by Title and Pclass
- `Embarked`: Filled with mode (most frequent value)
- `Fare`: Filled with median fare
- `Cabin`: Dropped due to high percentage of missing values

**Feature Engineering**:
- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, Master) from names
- **Family Size**: Combined SibSp and Parch to create total family size
- **IsAlone**: Binary feature indicating solo travelers
- **Age Bins**: Categorized ages into groups (Child, Teen, Adult, Middle, Senior)
- **Fare Bins**: Quartile-based fare categories

**Encoding**:
- One-hot encoding for categorical variables (Sex, Embarked, Title, etc.)
- Maintained consistency between train and test sets

### 3. Model Building

Tested three classification algorithms:

1. **Logistic Regression**: Simple baseline model
2. **Random Forest**: Ensemble method with decision trees
3. **Gradient Boosting**: Advanced boosting technique

**Training Strategy**:
- 80/20 train-validation split
- Stratified sampling to maintain class distribution
- 5-fold cross-validation for robustness

### 4. Model Evaluation

**Metrics Used**:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

### 5. Hyperparameter Tuning (Bonus)

Applied GridSearchCV to optimize Random Forest parameters:
- Number of estimators
- Maximum depth
- Minimum samples split/leaf

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | ~0.80 | ~0.78 | ~0.75 | ~0.76 | ~0.79 |
| Random Forest | ~0.83 | ~0.82 | ~0.78 | ~0.80 | ~0.82 |
| Gradient Boosting | ~0.82 | ~0.81 | ~0.77 | ~0.79 | ~0.81 |

*Note: Exact scores will vary based on random seed and data splits*

### Feature Importance

Top predictive features (Random Forest):
1. **Title_Mr**: Strong negative predictor (male passengers)
2. **Sex_male**: Gender was the strongest predictor
3. **Fare**: Higher fares correlated with survival
4. **Age**: Younger passengers had better survival rates
5. **Pclass**: First-class passengers survived more

## 💡 Key Insights

1. **Gender Bias**: Women had ~75% survival rate vs ~19% for men ("Women and children first" policy)

2. **Class Matters**: First-class passengers had ~63% survival rate vs ~24% for third-class

3. **Age Factor**: Children had higher survival rates regardless of class

4. **Family Dynamics**: Small families (2-4 members) had better survival rates than solo travelers or very large families

5. **Port of Embarkation**: Passengers from Cherbourg had slightly higher survival rates (likely correlated with class)

## 🔮 Future Improvements

1. **Advanced Feature Engineering**:
   - Extract deck information from cabin numbers
   - Create interaction features (e.g., Sex × Pclass)
   - Use passenger name lengths or rare names as features

2. **Model Enhancements**:
   - Try XGBoost or LightGBM
   - Implement ensemble methods (stacking, blending)
   - Neural networks for deep learning approach

3. **Data Augmentation**:
   - Use external historical data about Titanic
   - Impute missing values with more sophisticated methods (e.g., KNN imputation, MICE)

4. **Error Analysis**:
   - Deep dive into misclassified cases
   - Identify patterns in false positives/negatives

5. **Model Interpretability**:
   - SHAP values for better feature interpretation
   - Partial dependence plots

## 📚 Learning Outcomes

This project demonstrates:
- ✅ End-to-end machine learning workflow
- ✅ Data preprocessing and feature engineering techniques
- ✅ Multiple model comparison and selection
- ✅ Cross-validation and hyperparameter tuning
- ✅ Model evaluation with multiple metrics
- ✅ Understanding of classification bias and fairness

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by Kaggle's Titanic competition
- Inspired by the historical Titanic disaster and data science community
- Built as a learning project for machine learning classification

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Learning! 🚀**