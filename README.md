Gruppw22
Omran, Marwa  589105
Yekdaneh, Faranak
Pham, Tung
Hanitzsch, Richie


# 🏠 Berlin Housing Price Prediction

This project develops a machine learning pipeline to accurately predict housing prices in Berlin using a combination of property features, geographic location, demographic data, and advanced modeling techniques.

## 📌 Project Overview

- **Goal:** Predict residential property prices in Berlin.
- **Approach:** Use the CRISP-DM methodology (business understanding → data preparation → modeling → evaluation → deployment).
- **Models Used:** Linear Regression, Ridge, Lasso, Random Forest, XGBoost, and a Stacking Ensemble.
- **Best Performance:** R² = 0.9941 with Stacking Ensemble.

---

## 📊 Dataset

- **File Used:** `wohnungen_plus_burglary.csv`
- **Main Features:**
  - Area, rooms, construction year, heating type, floor
  - Borough, neighborhood, distance to city center
  - Median income, burglary rate, price per square meter

---

## ⚙️ Pipeline Structure

### 1. **Preprocessing**
- **Numerical Features:** KNN imputation + scaling
- **Categorical Features:** Mode imputation + OneHotEncoding
- **Location Features:** Custom handling for missing or inconsistent data

### 2. **Feature Engineering**
- Custom transformer to create:
  - Price per sqm
  - Distance category
  - Income-to-price ratios
- Automated transformations integrated into pipeline

### 3. **Model Pipeline**
- Preprocessing
- Feature engineering
- Stacking Ensemble with:
  - Base models: RandomForest, XGBoost, Ridge
  - Final estimator: GradientBoosting

---

## 📈 Model Evaluation

| Metric     | Value     |
|------------|-----------|
| R² Score   | 0.9941    |
| RMSE       | ~47,000 € |
| MAE        | ~29,000 € |

✅ All results meet or exceed business objectives.

---

## 🧠 Insights & Findings

- **Top Predictors:**
  - Area, rooms, distance to center, median income, neighborhood
- **Burglary Rate:** Slightly negatively correlated with price
- **District Variation:** Prices vary significantly across boroughs

---

## 🌍 Visualizations

- Interactive maps using **Folium**
- Dynamic charts with **Plotly**
- Heatmaps and distributions with **Seaborn**

---

## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt


   jupyter notebook berlin_housing_price_prediction.ipynb

  

