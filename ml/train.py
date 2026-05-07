import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

# LOAD DATA
df = pd.read_csv("dataset_A.csv")

 
# DATA CLEANING
 
# remove very short runs (buggy logs)
df = df[df["actual_travel_time"] > 3]

# remove invalid clearance
df = df[(df["avg_clearance"] > 0) & (df["min_clearance"] > 0)]

# drop NaNs just in case
df = df.dropna()

# FEATURE ENGINEERING
df["clearance_ratio"] = df["min_clearance"] / (df["avg_clearance"] + 1e-6)
df["curvature"] = df["smoothness"] * df["turn_count"]
df["log_uncertainty"] = np.log(df["uncertainty_trace"] + 1e-6)

#Clippings to handle outliers
df["log_uncertainty"] = df["log_uncertainty"].clip(-5, -1)
df["clearance_ratio"] = df["clearance_ratio"].clip(upper=1.5)

# FEATURE SELECTION
features = [
    "path_length",
    "smoothness",
    "turn_count",
    "avg_clearance",
    "min_clearance",
    "clearance_ratio",
    "curvature",
    "log_uncertainty",
]

# Encode categorical
df = pd.get_dummies(df, columns=["planner", "controller"])

features += [c for c in df.columns if c.startswith("planner_") or c.startswith("controller_")]

X = df[features]
y = df["inefficiency_factor"]

# Store times separately
ideal_time_all = df["ideal_travel_time"]
actual_time_all = df["actual_travel_time"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = {}

# TRAIN + TEST EVALUATION
for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    ideal_time = ideal_time_all.loc[X_test.index]
    actual_time = actual_time_all.loc[X_test.index]

    predicted_time = ideal_time * preds

    mae = mean_absolute_error(actual_time, predicted_time)
    rmse = np.sqrt(mean_squared_error(actual_time, predicted_time))

    results[name] = {
        "MAE (sec)": mae,
        "RMSE (sec)": rmse
    }

# CROSS VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}

for name, model in models.items():
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        ideal_time = ideal_time_all.iloc[test_idx]
        actual_time = actual_time_all.iloc[test_idx]

        predicted_time = ideal_time * preds

        mae = np.mean(np.abs(predicted_time - actual_time))
        fold_errors.append(mae)

    cv_results[name] = {
        "MAE per fold": fold_errors,
        "Mean MAE": np.mean(fold_errors),
        "Std Dev": np.std(fold_errors)
    }

# PRINT RESULTS
print("\n=== TEST SET PERFORMANCE ===")
for name, metrics in results.items():
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


# FEATURE IMPORTANCE (RF)
rf = models["Random Forest"]
importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
print(feat_imp)

# CROSS VALIDATION RESULTS
print("\n=== CROSS VALIDATION RESULTS ===")

for name, stats in cv_results.items():
    print(f"\n{name}")
    print("MAE per fold:", np.round(stats["MAE per fold"], 3))
    print("Mean MAE:", round(stats["Mean MAE"], 3))
    print("Std Dev:", round(stats["Std Dev"], 3))

# SAVE MODELS
gradient_boost = models["Gradient Boosting"]
decision_tree = models["Decision Tree"]
random_forest = models["Random Forest"]

gradient_boost.fit(X, y)
decision_tree.fit(X, y)
random_forest.fit(X, y)

joblib.dump(models["Gradient Boosting"], "models/gradient_boost_model.pkl")
joblib.dump(models["Decision Tree"], "models/decision_tree_model.pkl")
joblib.dump(models["Random Forest"], "models/random_forest_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("\nModels saved")