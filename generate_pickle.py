"""
generate_pickle.py
------------------
Reproduces the full preprocessing + training pipeline from credit_wise.ipynb
and saves all necessary artifacts as .pkl files for deployment.

Saved files:
  model.pkl          → Trained Logistic Regression model
  scaler.pkl         → Fitted StandardScaler
  label_encoder.pkl  → Fitted LabelEncoder (Education_Level & Loan_Approved)
  ohe_encoder.pkl    → Fitted OneHotEncoder (multi-column categorical cols)
  feature_columns.pkl→ Ordered list of feature column names expected by the model
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
df = pd.read_csv("loan_approval_data.csv")
print(f"✅ Data loaded — shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. Imputation (same strategy as notebook)
# ─────────────────────────────────────────────
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols   = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

print("✅ Imputation done")

# ─────────────────────────────────────────────
# 3. Drop ID column
# ─────────────────────────────────────────────
df = df.drop("Applicant_ID", axis=1)

# ─────────────────────────────────────────────
# 4. Label-encode ordinal / target columns
# ─────────────────────────────────────────────
le_education = LabelEncoder()
le_target    = LabelEncoder()

df["Education_Level"] = le_education.fit_transform(df["Education_Level"])
df["Loan_Approved"]   = le_target.fit_transform(df["Loan_Approved"])

print("✅ Label encoding done")
print(f"   Education classes : {list(le_education.classes_)}")
print(f"   Target classes    : {list(le_target.classes_)}")

# ─────────────────────────────────────────────
# 5. One-hot encode remaining categorical cols
# ─────────────────────────────────────────────
ohe_cols = ["Marital_Status", "Employment_Status", "Loan_Purpose",
            "Gender", "Employer_Category", "Property_Area"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded        = ohe.fit_transform(df[ohe_cols])
encoded_df     = pd.DataFrame(encoded,
                              columns=ohe.get_feature_names_out(ohe_cols),
                              index=df.index)

df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)
print("✅ One-hot encoding done")

# ─────────────────────────────────────────────
# 6. Feature engineering (squared terms — Cell 29)
# ─────────────────────────────────────────────
df["DTI_Ratio_sq"]    = df["DTI_Ratio"]    ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

feature_columns = list(X.columns)
print(f"✅ Feature engineering done — {len(feature_columns)} features")

# ─────────────────────────────────────────────
# 7. Train / test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 8. Scale
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Scaling done")

# ─────────────────────────────────────────────
# 9. Train model (Logistic Regression — best in notebook)
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\n📊 Model Performance on Test Set:")
print(f"   Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision : {precision_score(y_test, y_pred):.4f}")
print(f"   Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"   F1 Score  : {f1_score(y_test, y_pred):.4f}")

# ─────────────────────────────────────────────
# 10. Save all artifacts as pickle files
# ─────────────────────────────────────────────
artifacts = {
    "model.pkl"          : model,
    "scaler.pkl"         : scaler,
    "label_encoder.pkl"  : {"Education_Level": le_education, "Loan_Approved": le_target},
    "ohe_encoder.pkl"    : ohe,
    "feature_columns.pkl": feature_columns,
}

for filename, obj in artifacts.items():
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved → {filename}")

print("\n🎉 All pickle files generated successfully!")
print("\nPickle file summary:")
print("  model.pkl           → Logistic Regression classifier")
print("  scaler.pkl          → StandardScaler (fit on training data)")
print("  label_encoder.pkl   → Dict of LabelEncoders for Education_Level & Loan_Approved")
print("  ohe_encoder.pkl     → OneHotEncoder for categorical columns")
print("  feature_columns.pkl → Ordered list of feature names for inference")
