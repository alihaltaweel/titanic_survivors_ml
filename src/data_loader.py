import pandas as pd
import numpy as np

def preprocess_titanic_data(df):
    # 1. Feature Engineering: Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 2. Feature Engineering: Family Size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = 0
    df.loc[df['Family_Size'] == 1, 'Is_Alone'] = 1

    # 3. Handle Cabin (extract Deck)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # Drop columns that won't be used by the pipeline
    df = df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
    
    return df