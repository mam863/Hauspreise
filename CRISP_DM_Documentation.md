# Berlin Housing Price Prediction: CRISP-DM Process Documentation

This document outlines the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology applied to the Berlin housing price prediction project.

## 1. Business Understanding

### Business Objectives
- Develop a robust model to predict housing prices in Berlin based on property attributes
- Provide insights into key factors affecting property values in different Berlin neighborhoods
- Enable stakeholders to make informed decisions in the Berlin real estate market

### Key Stakeholders
- Real estate agents and agencies
- Property investors and developers
- Homebuyers and sellers
- Urban planners and policy makers

### Success Criteria
- Achieve a model with R² > 0.7 and RMSE < 15% of average property value
- Identify the most influential factors affecting housing prices
- Deliver actionable insights for stakeholders

## 2. Data Understanding

### Data Source
The dataset `wohnungen_mit_bezirk_excel.csv` contains information about housing properties in Berlin with the following attributes:

- **energy**: Type of energy used (Gas, Oel, Fernwaerme, etc.)
- **heating**: Heating system type
- **price**: Property price in Euros (target variable)
- **area**: Property area in square meters
- **rooms**: Number of rooms
- **zipcode**: Postal code
- **construction_year**: Year of construction
- **level**: Floor level
- **Country**: Country (all are in Germany)
- **City**: City (all are in Berlin)
- **Borough**: Berlin borough (e.g., Mitte, Pankow, etc.)
- **Neighborhood**: Specific neighborhood within the borough

### Data Exploration Findings
- The dataset contains approximately 5,000 property listings
- Price distribution is right-skewed with some luxury properties creating outliers
- Strong positive correlation between area and price
- Significant price variations across different boroughs and neighborhoods
- Some missing values in energy and heating columns
- Construction years range from historical buildings (1800s) to new developments (2024)

### Data Quality Issues
- Missing values in several columns (energy, heating)
- Some potentially incorrect construction years
- Possible outliers in price that may affect model performance

## 3. Data Preparation

### Data Cleaning
- Replaced "unbekannt" (unknown) values with NaN for proper handling
- Cleaned construction_year to remove implausible values
- Removed price outliers using IQR method to improve model robustness

### Feature Engineering
- Created 'building_age' feature (current year - construction_year)
- Calculated 'price_per_sqm' to analyze value metrics
- Preprocessed categorical features using one-hot encoding
- Standardized numerical features for better model performance

### Data Transformation
- Applied SimpleImputer to handle missing values:
  - Median strategy for numerical features
  - Most frequent value strategy for categorical features
- Used StandardScaler to normalize numerical features
- Created a preprocessing pipeline using ColumnTransformer

## 4. Modeling

### Modeling Techniques
We evaluated multiple regression algorithms:

1. **Random Forest Regressor**
   - Ensemble method resistant to overfitting
   - Handles non-linear relationships well
   - Provides feature importance metrics

2. **Gradient Boosting Regressor**
   - Sequential technique that builds on weak learners
   - Often achieves high accuracy
   - More sensitive to hyperparameters

3. **ElasticNet Regression**
   - Linear model with L1 and L2 regularization
   - Good for handling correlated features
   - More interpretable than tree-based models

4. **XGBoost Regressor**
   - Advanced implementation of gradient boosting
   - Highly optimized for performance
   - Generally provides excellent results for structured data

### Model Building Process
- Created scikit-learn pipelines combining preprocessing and model training
- Split data into 80% training and 20% test sets
- Ensured reproducibility with fixed random seed
- Trained each model with default parameters initially

## 5. Evaluation

### Model Performance Metrics
The models were evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score (coefficient of determination)

### Feature Importance Analysis
For the best-performing model:
- Identified the most influential features affecting housing prices
- Visualized top 20 features through importance plots
- Provided insights into how different factors impact property values

### Model Comparison
Detailed comparison of each model's performance metrics to identify the most effective approach for this specific problem.

## 6. Deployment

### Deployment Strategy
- Saved the best-performing model using joblib for persistence
- Created a simple prediction function for new data
- Generated visualizations and reports for business stakeholders

### Monitoring and Maintenance Plan
- Periodic retraining with new data (quarterly recommended)
- Performance monitoring using holdout validation sets
- Model version control for tracking changes

### Business Value Delivery
- Accurate price predictions to guide pricing strategies
- Insights into neighborhood-specific value drivers
- Identification of undervalued properties for investment

## Conclusion

This CRISP-DM process provided a structured approach to developing a predictive model for Berlin housing prices. The iterative nature of CRISP-DM allowed for continuous improvement and refinement of the model, resulting in actionable insights for stakeholders in the Berlin real estate market.

The documented process ensures transparency and reproducibility, allowing for future enhancements as more data becomes available or as market conditions change. 