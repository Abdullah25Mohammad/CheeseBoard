from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

data = pd.read_csv('data/training_data.csv')
X = data.drop(columns=['winner'])

def normalize(col):
    """
    Normalize a column by dividing by the maximum value in that column.
    """
    max_value = col.max()
    if max_value > 0:
        return col / max_value
    return col

# Normalize columns
for col in X.columns:
    X[col] = normalize(X[col])


y = data['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))



