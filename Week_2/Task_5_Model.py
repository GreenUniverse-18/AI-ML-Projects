import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) Load dataset
# -----------------------------
df = pd.read_csv("insurance.csv")

# -----------------------------
# 2) Transform (One-Hot Encoding for categoricals)
# -----------------------------
# Categorical columns: sex, smoker, region
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# -----------------------------
# 3) Split into X (features) and y (target)
# -----------------------------
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# -----------------------------
# 4) Train-test split (unseen test data)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------
# Helper: evaluate a regression model
# -----------------------------
def evaluate_regression_model(model, model_name: str):
    # Train
    model.fit(X_train, y_train)

    # Predict (train + test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    # Cross-validation (5-fold) using R²
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Overfitting / Underfitting indication (simple heuristic)
    gap = train_r2 - test_r2
    if gap > 0.10:
        fit_comment = "Likely overfitting (train >> test)"
    elif train_r2 < 0.50 and test_r2 < 0.50:
        fit_comment = "Likely underfitting (both low)"
    else:
        fit_comment = "Fit looks reasonable (train ~ test)"

    results = {
        "Model": model_name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "MAE": mae,
        "RMSE": rmse,
        "Test_R2_again": r2,  # same as Test_R2, kept for clarity
        "CV_R2_Mean": cv_mean,
        "CV_R2_Std": cv_std,
        "Fit_Diagnosis": fit_comment
    }

    return results


# -----------------------------
# 5) Instantiate models
# -----------------------------
lr_model = LinearRegression()

dt_model = DecisionTreeRegressor(
    random_state=42
    # You can control overfitting by setting:
    # max_depth=..., min_samples_split=..., min_samples_leaf=...
)

# -----------------------------
# 6) Evaluate both
# -----------------------------
lr_results = evaluate_regression_model(lr_model, "Linear Regression")
dt_results = evaluate_regression_model(dt_model, "Decision Tree Regressor")

# -----------------------------
# 7) Print results nicely
# -----------------------------
results_df = pd.DataFrame([lr_results, dt_results])

# Round for clean display
metrics_cols = ["Train_R2", "Test_R2", "MAE", "RMSE", "CV_R2_Mean", "CV_R2_Std"]
results_df[metrics_cols] = results_df[metrics_cols].applymap(lambda x: round(float(x), 4))

print("\n==================== MODEL COMPARISON (REGRESSION) ====================")
print(results_df[["Model", "Train_R2", "Test_R2", "MAE", "RMSE", "CV_R2_Mean", "CV_R2_Std", "Fit_Diagnosis"]])

# -----------------------------
# 8) Choose best model (example: highest CV mean R²)
best_idx = results_df["CV_R2_Mean"].astype(float).idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
print("\nBest model by mean CV R²:", best_model_name)

# 9) Predict medical charges for NEW unseen customers (example)
new_customers = pd.DataFrame([
    {"age": 29, "sex": "female", "bmi": 27.5, "children": 1, "smoker": "no",  "region": "northeast"},
    {"age": 52, "sex": "male",   "bmi": 33.2, "children": 2, "smoker": "yes", "region": "southeast"},
])

new_encoded = pd.get_dummies(new_customers, columns=["sex", "smoker", "region"], drop_first=True)
new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)

# Predict using both models
lr_pred_new = lr_model.predict(new_encoded)
dt_pred_new = dt_model.predict(new_encoded)

print("\n==================== NEW CUSTOMER PREDICTIONS ====================")
print(new_customers)
print("\nLinear Regression predicted charges:", np.round(lr_pred_new, 2))
print("Decision Tree predicted charges    :", np.round(dt_pred_new, 2))
