import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#LOad the dataset
df = pd.read_csv(r"C:\Users\uchen\Downloads\cmu-sleep.csv")

# Preprocessing

# Label Encoding of categorical variables
label_encoder = {}
for column in ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen', 'term_units', 'Zterm_units_ZofZ']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoder[column] = le




# Feature and target variable (independent and dependent variable)
# To define the feature variables
X = df[['TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep',
        'term_units', 'term_gpa']]

# To define the target variable
y = df['cum_gpa']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state= 40)

# Hyperparameter Tuning for Decision Tree
dt_params = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]

}

dt_grid_search = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5,
                              scoring='neg_mean_squared_error', error_score='raise')
dt_grid_search.fit(X_train, y_train)

dt_best_model = dt_grid_search.best_estimator_

print(f"Best Decision Tree Parameters: {dt_grid_search.best_params_}")

# Save the best Decision Tree model
joblib.dump(dt_best_model, 'decision_tree_model.joblib')

# Hyperparameter Tuning for Linear Regression
lr_params = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]

}

lr_grid_search = GridSearchCV(LinearRegression(), lr_params, cv=5,
                              scoring='neg_mean_squared_error', error_score='raise')
lr_grid_search.fit(X_train, y_train)

lr_best_model = lr_grid_search.best_estimator_

print(f"Best Linear Regression Parameters: {lr_grid_search.best_params_}")

# Save the best Linear Regression model
joblib.dump(lr_best_model, 'linear_regression_model.joblib')

# HyperTuning for Random Forest
rf_params = {
    'n_estimators': [5, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=5,
                              scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_

print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")

# Save the best Random Forest Model
joblib.dump(rf_best_model, 'random_forest_model.joblib')

#Train and Evaluate models with scaled features
models = {
    'Linear Regression': lr_best_model,
    'Decision Tree': dt_best_model,
    'Random Forest': rf_best_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - Mean Squared Error: {mse}')
    print(f'{name} - R2 Score: {r2}')

# Save Model
joblib.dump(model, f'{name.lower().replace( " ", "_")}_model.joblib')
