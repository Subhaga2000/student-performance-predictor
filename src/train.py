# src/train.py
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === Step 1: Load dataset ===
data_path = Path("data/student-mat.csv")
df = pd.read_csv(data_path, sep=';')

# === Step 2: Rename columns for Sri Lanka context ===
df = df.rename(columns={
    "age": "Age",
    "famsize": "Family_Size",
    "Medu": "Mother_Education",
    "Mjob": "Mother_Job",
    "Fedu": "Father_Education",
    "Fjob": "Father_Job",
    "reason": "School_Reason",
    "traveltime": "Travel_Time",
    "studytime": "Study_Time",
    "failures": "Past_Failures",
    "schoolsup": "Tuition_Support",
    "famsup": "Family_Support",
    "paid": "Private_Classes",
    "activities": "Extra_Activities",
    "nursery": "Nursery_Education",
    "higher": "University_Plan",
    "internet": "Internet_Access",
    "romantic": "Relationship_Status",
    "famrel": "Family_Relation",
    "freetime": "Free_Time_After_School",
    "goout": "Social_Time",
    "Dalc": "Weekday_Alcohol",
    "Walc": "Weekend_Alcohol",
    "health": "Health_Status",
    "absences": "School_Absences",
    "G1": "Term1_Marks",
    "G2": "Term2_Marks",
    "G3": "Final_Marks"
})

# === Step 3: Create binary target (pass/fail) ===
df['passed'] = (df['Final_Marks'] >= 10).astype(int)

# === Step 4: Features & target ===
X = df.drop(columns=['Final_Marks', 'passed'])
y = df['passed']

# Identify numeric & categorical columns automatically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# === Step 5: Preprocessing ===
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# === Step 6: Full pipeline with RandomForestClassifier ===
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

# === Step 7: Train/test split (stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 8: Hyperparameter tuning (GridSearch) ===
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': ['balanced', None]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# === Step 9: Evaluate best model ===
best = grid.best_estimator_
print("Best params:", grid.best_params_)

y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# === Step 10: Save model & metadata ===
Path("models").mkdir(exist_ok=True)
joblib.dump(best, "models/student_model.pkl")

# Save features & default values for web app input form
features = X.columns.tolist()
joblib.dump(features, "models/features.pkl")

defaults = {}
for col in features:
    if col in numeric_features:
        defaults[col] = float(X[col].median())
    else:
        defaults[col] = str(X[col].mode()[0])

with open("models/defaults.json", "w") as f:
    json.dump(defaults, f, indent=2)

print("Saved model & metadata to models/")
