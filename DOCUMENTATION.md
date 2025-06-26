# Berlin Housing Price Prediction Model Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Model Performance](#model-performance)
7. [Usage Guide](#usage-guide)
8. [Maintenance and Updates](#maintenance-and-updates)

## Project Overview

### Purpose
This project develops a machine learning model to predict housing prices in Berlin, Germany. The model uses various property characteristics, location data, and engineered features to provide accurate price estimates for residential properties.

### Key Features
- Stacking ensemble architecture combining multiple algorithms
- Comprehensive feature engineering
- Robust handling of location-based information
- High prediction accuracy (R² = 0.9941)

## Data Description

### Source Dataset
- Primary dataset: 'wohnungen_mit_korrekter_entfernung.csv'
- Contains Berlin housing market data with property characteristics and location information

### Key Variables
1. **Property Characteristics:**
   - Area (m²)
   - Number of rooms
   - Construction year
   - Floor level
   - Energy efficiency rating
   - Heating type

2. **Location Information:**
   - Borough (District)
   - Neighborhood
   - Distance to city center
   - Zipcode

## Data Preprocessing

### Cleaning Steps
1. **Missing Value Treatment:**
   - KNN imputation for numerical features
   - Mode imputation for categorical features
   - Custom handling for location-based missing data

2. **Outlier Handling:**
   - IQR-based outlier detection
   - Domain-specific validation rules
   - Careful preservation of legitimate high-value properties

3. **Data Type Conversions:**
   - Standardized numeric formats
   - Categorical encoding
   - Date/time formatting

## Feature Engineering

### Created Features

1. **Time-based Features:**
   - Building age
   - Is new building (≤5 years)
   - Needs renovation (>30 years)

2. **Area-based Features:**
   - Rooms per area ratio
   - Is spacious (based on median room ratio)
   - Area-room interaction

3. **Location-based Features:**
   - Location score (based on borough statistics)
   - Location clusters (K-means, n=5)
   - Borough price levels
   - Distance-based features

4. **Price-related Features:**
   - Price per square meter
   - Zipcode price statistics
   - Borough price statistics

5. **Energy and Heating Features:**
   - Heating efficiency score
   - Energy cost indicator
   - Combined efficiency rating

### Feature Importance
Top influential features (in order of importance):
1. Area
2. Location score
3. Distance to center
4. Building age
5. Room count

## Model Architecture

### Stacking Ensemble

1. **Base Models:**
   - Random Forest (n_estimators=200)
   - XGBoost
   - LightGBM
   - Gradient Boosting

2. **Meta-learner:**
   - Ridge Regression

3. **Cross-validation:**
   - 5-fold cross-validation
   - Stratified by price ranges

### Pipeline Components

1. **Preprocessor:**
   ```python
   preprocessor = ColumnTransformer([
       ('num', numeric_transformer, numeric_features),
       ('cat', categorical_transformer, categorical_features)
   ])
   ```

2. **Feature Engineering:**
   - Custom transformer for advanced features
   - Automated feature creation pipeline

3. **Model Pipeline:**
   - Preprocessing
   - Feature engineering
   - Stacking ensemble
   - Final prediction

## Model Performance

### Metrics
- RMSE: 17,941.30 €
- MAE: 12,151.83 €
- R² Score: 0.9941

### Performance Analysis
1. **Error Distribution:**
   - Symmetric error distribution
   - Consistent performance across price ranges
   - Slightly higher variance in luxury segment

2. **Cross-validation Results:**
   - Stable performance across folds
   - No significant overfitting observed
   - Robust to different data splits

## Usage Guide

### Basic Usage
```python
from berlin_housing_model import predict_house_price

# Example property
example_house = {
    'area': 85.0,
    'rooms': 3,
    'construction_year': 1990,
    'level': 2,
    'energy': 'B',
    'heating': 'Fernwärme',
    'Borough': 'Mitte',
    'Neighborhood': 'Wedding',
    'zipcode': 13355,
    'distance_to_center_km': 2.5
}

# Get prediction
price = predict_house_price(example_house)
```

### Required Features
All input features must be provided:
- Numerical features (as float/int)
- Categorical features (as strings)
- Location information (Borough, Neighborhood, zipcode)

### Model Files
- Main model: 'berlin_housing_model.pkl'
- Feature engineering pipeline: included in model file
- Requirements: see requirements.txt

## Maintenance and Updates

### Recommended Update Frequency
- Retrain model every 3-6 months
- Update feature engineering as market changes
- Monitor prediction accuracy regularly

### Monitoring
1. **Track Metrics:**
   - Prediction accuracy
   - Feature importance shifts
   - Error patterns

2. **Data Quality:**
   - Input data validation
   - Feature distribution monitoring
   - Missing value patterns


---

## Contact and Support
For questions and support, please refer to the project repository or contact the development team.

Last Updated: [Current Date]
Version: 1.0 