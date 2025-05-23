import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('AeroReach Insights.csv')

# Check data
print(data.head())
print(data.info())


# Check for missing values
print(data.isnull().sum())

# Convert target variable to numeric
data['Taken_product'] = data['Taken_product'].map({'Yes':1, 'No':0})

# Encode categorical variables
categorical_cols = ['preferred_device', 'preferred_location_type', 'following_company_page', 'working_flag', 'Adult_flag']

for col in categorical_cols:
    data[col] = data[col].astype(str)  # ensure string type
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

print(data.head())

X = data.drop(['UserID', 'Taken_product'], axis=1)  # drop UserID and target from features
y = data['Taken_product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Identify columns with non-numeric entries
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Non-numeric entries in column {col}:")
        print(X[col].unique())


# Convert non-numeric entries to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the median of each column
X.fillna(X.median(), inplace=True)

# Drop rows with missing values
X.dropna(inplace=True)

# Convert all columns to numeric
X = X.apply(pd.to_numeric)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))


# 5th

importances = rf.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance - Random Forest")
plt.show()


#6 

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba_xgb))



import os
print(f"Current working directory: {os.getcwd()}")

with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model saved successfully as model.pkl")
print("Files in current directory:", os.listdir())


