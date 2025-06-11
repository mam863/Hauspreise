"""
Berlin Housing Price Prediction - Model Enhancement
Advanced feature engineering and hyperparameter tuning

This script builds on the base model to enhance performance through:
1. Advanced feature engineering
2. Hyperparameter tuning
3. Model stacking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import os
from scipy.stats import randint, uniform

# Load the data
print("Loading dataset...")
data = pd.read_csv('wohnungen_mit_bezirk_excel.csv', sep=';')

# Advanced Feature Engineering
print("\nPerforming advanced feature engineering...")

# Replace 'unbekannt' with NaN
data['heating'].replace('unbekannt', np.nan, inplace=True)
data['energy'].replace('unbekannt', np.nan, inplace=True)

# Clean construction_year
def clean_year(year):
    if pd.isna(year):
        return np.nan
    if year < 1800 or year > 2025:
        return np.nan
    return year

data['construction_year'] = data['construction_year'].apply(clean_year)

# 1. Neighborhood-based features
# Calculate neighborhood stats
neighborhood_price_mean = data.groupby('Neighborhood')['price'].mean()
neighborhood_price_median = data.groupby('Neighborhood')['price'].median()
neighborhood_price_std = data.groupby('Neighborhood')['price'].std()
neighborhood_price_count = data.groupby('Neighborhood')['price'].count()

# Add neighborhood stats as features
data['neighborhood_price_mean'] = data['Neighborhood'].map(neighborhood_price_mean)
data['neighborhood_price_median'] = data['Neighborhood'].map(neighborhood_price_median)
data['neighborhood_price_std'] = data['Neighborhood'].map(neighborhood_price_std)
data['neighborhood_density'] = data['Neighborhood'].map(neighborhood_price_count)

# 2. Building age-related features
current_year = 2023
data['building_age'] = current_year - data['construction_year']

# Age categories (historical significance)
data['is_prewar'] = (data['construction_year'] < 1939).astype(int)
data['is_postwar'] = ((data['construction_year'] >= 1945) & (data['construction_year'] <= 1970)).astype(int)
data['is_modern'] = (data['construction_year'] > 1990).astype(int)
data['is_new'] = (data['construction_year'] > 2010).astype(int)

# 3. Price per square meter
data['price_per_sqm'] = data['price'] / data['area']

# 4. Room density
data['room_density'] = data['rooms'] / data['area']

# 5. Floor level features
# High floor premium (top 20% of floors)
floor_percentile_80 = data['level'].quantile(0.8)
data['is_high_floor'] = (data['level'] >= floor_percentile_80).astype(int)
data['is_ground_floor'] = (data['level'] == 1).astype(int)

# 6. Zipcode clustering - group similar areas
# Extract first 2 digits of zipcode to get district
data['district'] = data['zipcode'].astype(str).str[:2]

# 7. Heating and energy categories
data['has_central_heating'] = data['heating'].str.contains('Zentralheizung', na=False).astype(int)
data['has_floor_heating'] = data['heating'].str.contains('Fussbodenheizung', na=False).astype(int)
data['has_gas'] = (data['energy'] == 'Gas').astype(int)
data['has_remote_heat'] = (data['energy'] == 'Fernwaerme').astype(int)

# 8. Location-based feature: distance to city center
# For simplicity, we'll use zipcode as a proxy for distance
# Berlin city center is approximately 10117
def zipcode_distance(zipcode, center=10117):
    try:
        return abs(int(zipcode) - center)
    except:
        return np.nan

data['distance_to_center'] = data['zipcode'].apply(zipcode_distance)

# 9. Interaction features
data['area_per_room'] = data['area'] / data['rooms']
data['age_area_interaction'] = data['building_age'] * data['area']

# Clean up outliers
print("\nCleaning outliers...")
# Price outliers
Q1_price = data['price'].quantile(0.01)
Q3_price = data['price'].quantile(0.99)
IQR_price = Q3_price - Q1_price
data = data[(data['price'] >= Q1_price - 1.5 * IQR_price) & (data['price'] <= Q3_price + 1.5 * IQR_price)]

# Area outliers
Q1_area = data['area'].quantile(0.01)
Q3_area = data['area'].quantile(0.99)
IQR_area = Q3_area - Q1_area
data = data[(data['area'] >= Q1_area - 1.5 * IQR_area) & (data['area'] <= Q3_area + 1.5 * IQR_area)]

# Split features and target
X = data.drop(['price', 'price_per_sqm'], axis=1)
y = data['price']

# Define feature groups for preprocessing
categorical_features = ['energy', 'heating', 'Borough', 'Neighborhood', 'district']
numerical_features = [
    'area', 'rooms', 'construction_year', 'level', 'building_age',
    'neighborhood_price_mean', 'neighborhood_price_median', 'neighborhood_price_std',
    'neighborhood_density', 'room_density', 'distance_to_center',
    'area_per_room', 'age_area_interaction'
]
binary_features = [
    'is_prewar', 'is_postwar', 'is_modern', 'is_new',
    'is_high_floor', 'is_ground_floor', 'has_central_heating',
    'has_floor_heating', 'has_gas', 'has_remote_heat'
]

# Advanced preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# No transformation needed for binary features
binary_transformer = 'passthrough'

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
print("\nPerforming hyperparameter tuning...")

# 1. Random Forest Tuning
rf_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

rf_grid_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

print("Tuning Random Forest...")
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
rf_best_params = rf_grid_search.best_params_
print(f"Best Random Forest Parameters: {rf_best_params}")

# 2. XGBoost Tuning
xgb_param_grid = {
    'model__n_estimators': randint(50, 300),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__colsample_bytree': uniform(0.5, 0.5),
    'model__gamma': uniform(0, 1),
    'model__min_child_weight': randint(1, 10)
}

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(random_state=42))
])

xgb_grid_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

print("Tuning XGBoost...")
xgb_grid_search.fit(X_train, y_train)
xgb_best_model = xgb_grid_search.best_estimator_
xgb_best_params = xgb_grid_search.best_params_
print(f"Best XGBoost Parameters: {xgb_best_params}")

# 3. Gradient Boosting Tuning
gb_param_grid = {
    'model__n_estimators': randint(50, 300),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__min_samples_split': randint(2, 10),
    'model__min_samples_leaf': randint(1, 10)
}

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

gb_grid_search = RandomizedSearchCV(
    gb_pipeline,
    param_distributions=gb_param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

print("Tuning Gradient Boosting...")
gb_grid_search.fit(X_train, y_train)
gb_best_model = gb_grid_search.best_estimator_
gb_best_params = gb_grid_search.best_params_
print(f"Best Gradient Boosting Parameters: {gb_best_params}")

# 4. ElasticNet Tuning
en_param_grid = {
    'model__alpha': uniform(0.001, 1.0),
    'model__l1_ratio': uniform(0, 1),
    'model__max_iter': [1000, 2000, 3000]
}

en_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ElasticNet(random_state=42))
])

en_grid_search = RandomizedSearchCV(
    en_pipeline,
    param_distributions=en_param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

print("Tuning ElasticNet...")
en_grid_search.fit(X_train, y_train)
en_best_model = en_grid_search.best_estimator_
en_best_params = en_grid_search.best_params_
print(f"Best ElasticNet Parameters: {en_best_params}")

# Model Stacking
print("\nBuilding stacked ensemble model...")

# Define base models with tuned hyperparameters
base_models = [
    ('rf', rf_best_model),
    ('xgb', xgb_best_model),
    ('gb', gb_best_model),
    ('en', en_best_model)
]

# Define meta-learner
meta_learner = Ridge(alpha=0.5)

# Create stacking regressor
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

# Fit stacked model
stacked_model.fit(X_train, y_train)

# Evaluate all models
print("\nEvaluating models...")

# Create directory for model results
if not os.path.exists('enhanced_model_results'):
    os.makedirs('enhanced_model_results')

# Function to evaluate and save model results
def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Save model
    joblib.dump(model, f'enhanced_model_results/{name.lower().replace(" ", "_")}.pkl')
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{name}: Actual vs Predicted Prices')
    plt.savefig(f'enhanced_model_results/{name.lower().replace(" ", "_")}_predictions.png')
    
    return {
        'name': name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'model': model
    }

# Evaluate all models
models = [
    ('Random Forest (Tuned)', rf_best_model),
    ('XGBoost (Tuned)', xgb_best_model),
    ('Gradient Boosting (Tuned)', gb_best_model),
    ('ElasticNet (Tuned)', en_best_model),
    ('Stacked Ensemble', stacked_model)
]

results = []
for name, model in models:
    result = evaluate_model(model, name, X_train, X_test, y_train, y_test)
    results.append(result)

# Find best model
best_model_result = min(results, key=lambda x: x['test_rmse'])
print(f"\nBest Model: {best_model_result['name']}")
print(f"Test RMSE: {best_model_result['test_rmse']:.2f}")
print(f"Test R²: {best_model_result['test_r2']:.4f}")

# Save best model
joblib.dump(best_model_result['model'], 'enhanced_best_model.pkl')

# Feature importance for best model (if available)
if best_model_result['name'] != 'Stacked Ensemble' and best_model_result['name'] != 'ElasticNet (Tuned)':
    model = best_model_result['model']
    
    # Get preprocessor and model components
    preprocessor = model.named_steps['preprocessor']
    model_component = model.named_steps['model']
    
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback
        feature_names = ['feature_' + str(i) for i in range(preprocessor.transform(X_test.iloc[[0]]).shape[1])]
    
    # Get feature importances
    if hasattr(model_component, 'feature_importances_'):
        importances = model_component.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model_result["name"]}')
        plt.tight_layout()
        plt.savefig('enhanced_model_results/enhanced_feature_importance.png')
        
        # Save feature importances
        feature_importance.to_csv('enhanced_model_results/feature_importances.csv', index=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))

print("\nEnhanced model development complete!") 