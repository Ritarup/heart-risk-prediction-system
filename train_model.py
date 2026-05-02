import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('cardio_train.csv', sep=';')
data = data.drop(columns=['id'])
data['age'] = (data['age'] / 365).astype(int)

X = data.drop('cardio', axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl', compress=9)