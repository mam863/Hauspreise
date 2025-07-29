# Berlin Housing Price Prediction Model
# Segmentation Analysis Documentation

## Overview

This document provides a detailed analysis of the Berlin Housing Price prediction model performance across different market segments. The segmentation analysis aims to identify specific areas where the model excels and where it struggles, enabling targeted improvements and more reliable price predictions for different types of properties.

## Methodology

The analysis evaluates model performance using:
- **Absolute Error**: The absolute difference between predicted and actual prices (in €)
- **Percentage Error**: The absolute error as a percentage of the actual price
- **Mean and Median**: Both metrics are used to account for skewed distributions

The dataset was segmented across five key dimensions:
1. Price Category
2. Property Size
3. Property Age
4. Borough (Location)
5. Distance from City Center

## Key Findings

### Price Category Performance

| Price Category | Mean % Error | Mean Absolute Error (€) | Sample Count |
|----------------|--------------|-------------------------|--------------|
| Budget (<250K €) | 14.40% | 26,546 | 184 |
| Mid-range (250K-500K €) | 7.33% | 26,512 | 418 |
| Premium (500K-1M €) | 6.95% | 49,654 | 234 |
| Luxury (>1M €) | 9.65% | 145,526 | 85 |

**Insights:**
- Budget properties have the highest percentage error despite having lower absolute errors
- Premium properties show the best predictive accuracy in percentage terms
- Luxury properties have extremely high absolute errors but relatively moderate percentage errors
- Mid-range properties benefit from largest sample size and show good predictive performance

### Property Size Impact

| Size Category | Mean % Error | Mean Absolute Error (€) | Sample Count |
|---------------|--------------|-------------------------|--------------|
| Very Small | 11.35% | 21,988 | 73 |
| Small | 8.20% | 25,928 | 157 |
| Medium | 6.93% | 32,216 | 103 |
| Large | 7.90% | 62,608 | 77 |
| Very Large | 12.78% | 83,033 | 106 |

**Insights:**
- Medium-sized properties show the best predictive accuracy
- Very Small and Very Large properties exhibit significantly higher percentage errors
- Error distribution forms a U-shaped pattern across size categories
- Absolute errors increase with property size, as expected

### Property Age Analysis

| Age Category | Mean % Error | Mean Absolute Error (€) | Sample Count |
|--------------|--------------|-------------------------|--------------|
| New (0-20 years) | 11.96% | 94,053 | 83 |
| Middle-aged (21-50 years) | 8.96% | 43,091 | 133 |
| Old (51+ years) | 8.79% | 37,989 | 640 |

**Insights:**
- Newer properties are significantly harder to predict accurately
- Older properties show better predictive performance
- Most properties in the dataset are older (51+ years)
- Both percentage and absolute errors decrease with property age

### Borough Analysis (Top 5 Highest Errors)

| Borough | Mean % Error | Mean Absolute Error (€) | Sample Count |
|---------|--------------|-------------------------|--------------|
| Neukölln | 9.79% | 44,402 | 50 |
| Steglitz-Zehlendorf | 9.57% | 50,658 | 102 |
| Mitte | 9.34% | 46,237 | 115 |
| Friedrichshain-Kreuzberg | 9.19% | 62,077 | 102 |
| Charlottenburg-Wilmersdorf | 7.66% | 45,395 | 146 |

**Insights:**
- Neukölln shows the highest percentage error, possibly due to rapid gentrification
- Central districts and those undergoing significant transformation are harder to predict
- Boroughs with more stable markets show better predictive performance
- Friedrichshain-Kreuzberg has the highest absolute error, reflecting its expensive and volatile market

### Location Category

Insufficient data available across location categories in the test set to draw reliable conclusions.

## Visualization Insights

The boxplot visualizations reveal:

1. **Price Categories**:
   - Budget properties have the widest spread in percentage errors
   - Premium properties show the most consistent predictions
   - Luxury properties have extreme outliers in absolute terms

2. **Property Size**:
   - Large and Very Large properties show higher variance in prediction errors
   - Medium properties have the narrowest error distribution
   - Very Small properties have concentrated errors but some extreme outliers

3. **Property Age**:
   - Newer properties show wider error distributions
   - Old properties have more outliers but better median performance
   - Middle-aged properties have the most consistent predictions

4. **Borough Performance**:
   - Top 5 boroughs with highest errors show similar error patterns (7-10%)
   - Charlottenburg-Wilmersdorf shows better performance than other central districts

## Model Strengths and Weaknesses

### Model Performs Best With:
- Premium properties (500K-1M €)
- Medium-sized properties
- Older properties (51+ years)
- Properties in Charlottenburg-Wilmersdorf

### Model Struggles Most With:
- Budget properties under 250K €
- Very Large and Very Small properties
- Newly built properties (0-20 years old)
- Properties in Neukölln and Steglitz-Zehlendorf

## Recommendations for Model Improvement

1. **Budget Properties Focus**:
   - Create separate models or apply boosting techniques for budget properties
   - Incorporate additional features specific to low-price segment
   - Consider more neighborhood-level variables for budget properties

2. **Size-Specific Enhancements**:
   - Develop specialized feature sets for Very Small and Very Large properties
   - Add interaction terms between size and other features
   - Consider non-linear transformations of the area variable

3. **New Construction Handling**:
   - Add more features related to building materials and quality for newer properties
   - Incorporate neighborhood development trends for new construction areas
   - Consider time-based features to capture market trends for newer properties

4. **Borough-Specific Adjustments**:
   - Develop borough-specific models for high-error areas
   - Add more granular location data, especially for rapidly changing neighborhoods
   - Include gentrification indicators for areas like Neukölln

5. **Data Enrichment Opportunities**:
   - Add public transit accessibility metrics
   - Include school quality and amenity proximity data
   - Add socioeconomic indicators at neighborhood level
   - Include historical price trends by neighborhood

## Technical Implementation Notes

To implement these improvements, consider:

1. **Ensemble Approach**: Create separate models for each segment and combine predictions
2. **Hierarchical Modeling**: Use nested models that first identify segment, then predict within segment
3. **Feature Engineering**: Develop segment-specific feature sets
4. **Sampling Techniques**: Use oversampling for underrepresented but high-error segments
5. **Hyperparameter Optimization**: Tune model parameters separately for each identified segment

## Conclusion

The segmentation analysis reveals that while the model performs well overall, there are specific segments where accuracy could be substantially improved. Budget properties, very large/small properties, and newer constructions present the greatest opportunities for model enhancement. By focusing on these segments, we can develop a more robust and accurate prediction model for Berlin's diverse housing market.

---


**Analysis By**: Marwa Omran
