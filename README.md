# Berlin Housing Price Prediction Model

This project implements a machine learning model to predict housing prices in Berlin following the CRISP-DM methodology. The model uses various property features like area, rooms, location, and building characteristics to predict prices.

## Project Structure

- `berlin_housing_model.py`: Main model implementation following CRISP-DM process
- `model_enhancement.py`: Advanced feature engineering and model tuning
- `model_deployment.py`: Flask API for serving model predictions
- `CRISP_DM_Documentation.md`: Detailed documentation of the CRISP-DM process
- `requirements.txt`: Required Python packages
- `wohnungen_mit_bezirk_excel.csv`: Dataset containing Berlin housing data

## Features

- Complete implementation of the CRISP-DM methodology
- Advanced feature engineering with neighborhood statistics
- Multiple model comparison (Random Forest, XGBoost, Gradient Boosting, ElasticNet)
- Model stacking for improved performance
- Hyperparameter tuning
- Visualization of results and feature importance
- Robust evaluation metrics (RMSE, MAE, R²)
- REST API for model deployment

## CRISP-DM Process

This project follows the Cross-Industry Standard Process for Data Mining:

1. **Business Understanding**: Define objectives and success criteria
2. **Data Understanding**: Explore and visualize the dataset
3. **Data Preparation**: Clean data, handle missing values, engineer features
4. **Modeling**: Build and compare multiple models
5. **Evaluation**: Assess model performance with metrics
6. **Deployment**: Create prediction API and deployment artifacts

See `CRISP_DM_Documentation.md` for detailed documentation of each phase.

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Model

Run the basic model implementation:

```bash
python berlin_housing_model.py
```

This will:
- Load and explore the dataset
- Prepare the data with basic preprocessing
- Train multiple models
- Evaluate and compare model performance
- Save the best model as `best_model.pkl`

### Enhanced Model

Run the enhanced model with advanced features:

```bash
python model_enhancement.py
```

This will:
- Perform advanced feature engineering
- Tune hyperparameters for all models
- Create a stacked ensemble model
- Evaluate all models and compare performance
- Save the best model as `enhanced_best_model.pkl`

### Model Deployment API

Deploy the model as a REST API:

```bash
python model_deployment.py
```

This will start a Flask server on port 5000 with the following endpoints:

- **POST /predict**: Make a prediction with the model
- **GET /health**: Check if the model is loaded and ready
- **GET /metadata**: Get information about the model and required features

#### Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 80.0,
    "rooms": 3.0,
    "construction_year": 2000.0,
    "level": 2.0,
    "energy": "Gas",
    "heating": "Zentralheizung",
    "Borough": "Mitte",
    "Neighborhood": "Mitte"
  }'
```

#### Example API Response

```json
{
  "predicted_price": 450000.25,
  "confidence_interval": {
    "lower": 405000.23,
    "upper": 495000.28
  },
  "currency": "EUR",
  "property": {
    "area": 80.0,
    "rooms": 3.0,
    "construction_year": 2000.0,
    "level": 2.0,
    "energy": "Gas",
    "heating": "Zentralheizung",
    "Borough": "Mitte",
    "Neighborhood": "Mitte"
  }
}
```

### Making Predictions in Python

To make predictions directly with the model:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('enhanced_best_model.pkl')

# Example property data
property_data = {
    'area': 80.0,
    'rooms': 3.0,
    'construction_year': 2000.0,
    'level': 2.0,
    'energy': 'Gas',
    'heating': 'Zentralheizung',
    'Borough': 'Mitte',
    'Neighborhood': 'Mitte',
    # Other features will be handled by the preprocessing pipeline
}

# Convert to DataFrame
input_df = pd.DataFrame([property_data])

# Make prediction
predicted_price = model.predict(input_df)[0]
print(f"Predicted price: €{predicted_price:.2f}")
```

## Results

The models are evaluated based on:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

Model comparison results and visualizations are saved in the `model_results/` and `enhanced_model_results/` directories.

## Feature Importance

Feature importance analysis reveals which property characteristics have the strongest impact on price predictions. This information can be valuable for:
- Real estate agents determining key selling points
- Property developers deciding which features to prioritize
- Homebuyers understanding value drivers in the Berlin market

## License

This project is licensed under the MIT License - see the LICENSE file for details. 