from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def build_and_tune_model(X_train, y_train):
    # 1. Define Preprocessing for Numeric Data
    numeric_features = ['Age', 'Fare', 'Family_Size']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Define Preprocessing for Categorical Data
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combine into a Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Create the Full Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 5. Define Hyperparameters to Tune (The "Mid-Level" part)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }

    # 6. Grid Search with Cross-Validation
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_