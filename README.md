# Student Lifestyle & Academic Performance Analysis

A machine learning investigation into how lifestyle factors influence student academic success, featuring a comparative study of four regression models enhanced with hyperparameter tuning.

## Project Overview

This project investigates how various lifestyle factors impact student academic performance through predictive modeling. By analyzing sleep patterns, study habits, social activities, and stress levels, the study provides insights into optimal lifestyle choices for academic success. The project implements and compares four different machine learning approaches to understand the underlying relationships between lifestyle and grades.

## Architecture

```
Lifestyle Data → Feature Engineering → Model Training → Hyperparameter Tuning → Performance Comparison → Insights
```

## Technology Stack

**Machine Learning & Data Science:**
- Python (scikit-learn, pandas, NumPy)
- Linear Regression, Decision Trees, Random Forest, K-Nearest Neighbors
- GridSearchCV for hyperparameter optimization

**Data Analysis:**
- Statistical analysis and correlation studies
- Feature importance evaluation
- Model performance metrics (R-squared, MSE)

## Dataset Overview

**Data Source**: [Student Lifestyle Factors Dataset](https://www.kaggle.com/datasets/charlottebennett1234/lifestyle-factors-and-their-impact-on-students)

**Target Variable:**
- Student grades (continuous)

**Predictor Variables:**
- Study hours per day
- Extracurricular hours per day
- Sleep hours per day
- Social hours per day
- Physical activity hours per day
- Stress level
- Gender

## Model Performance Comparison

| Model | R-squared Score | MSE | Optimal Parameters | Key Insights |
|-------|----------------|-----|-------------------|--------------|
| **Linear Regression** | **0.52** | **0.25** | N/A | Baseline linear relationship |
| Decision Tree | 0.51 | 0.26 | max_depth=3 | Non-linear pattern detection |
| Random Forest | 0.51 | 0.25 | max_depth=3 | Ensemble robustness |
| K-Nearest Neighbors | 0.50 | 0.26 | n_neighbors=28 | Local pattern recognition |

## Technical Implementation

### Linear Regression Analysis (student_lifestyle_linreg.ipynb)
- **Model Type**: Ordinary Least Squares regression
- **Performance**: R² = 0.52, MSE = 0.25
- **Key Finding**: Establishes baseline linear relationship between lifestyle factors and grades
- **Feature Analysis**: Direct coefficient interpretation for lifestyle impact

### Decision Tree Modeling (student_lifestyle_decisiontree.ipynb)
- **Optimization**: GridSearchCV for optimal tree depth
- **Best Parameters**: max_depth = 3
- **Performance**: R² = 0.51, MSE = 0.26

### Random Forest Implementation (student_lifestyle_randomforest.ipynb)
- **Ensemble Method**: Multiple decision trees with aggregation
- **Configuration**: max_depth = 3 for consistency
- **Performance**: R² = 0.51, MSE = 0.25

### K-Nearest Neighbors Modeling (student_lifestyle_knn.ipynb)
- **Optimization**: GridSearchCV for neighbor count
- **Best Parameters**: n_neighbors = 28
- **Performance**: R² = 0.50, MSE = 0.26

## Key Findings & Insights

### Model Performance Analysis
- **Consistent Performance**: All models achieve similar R² scores (0.50-0.52)
- **Linear Relationship Indication**: Minimal performance difference suggests underlying linearity
- **Feature Sufficiency**: Current features explain ~52% of grade variance
- **Model Complexity**: Simple models perform as well as complex ones

### Lifestyle Impact Insights
- **Study Hours**: Primary predictor of academic performance
- **Sleep Quality**: Significant correlation with grade outcomes
- **Stress Management**: Inverse relationship with academic success
- **Work-Life Balance**: Optimal distribution of time across activities
- **Physical Activity**: Positive correlation with cognitive performance

### Statistical Implications
- **Linear Relationships**: Predominant linear associations between predictors and grades
- **Data Limitations**: Performance plateau suggests additional features needed
- **Multicollinearity**: Potential interactions between lifestyle factors
- **Generalizability**: Model consistency indicates robust patterns

## Project Structure

```
├── student_lifestyle_linreg.ipynb       # Linear regression implementation
├── student_lifestyle_decisiontree.ipynb # Decision tree analysis
├── student_lifestyle_randomforest.ipynb # Random forest modeling
├── student_lifestyle_knn.ipynb          # K-nearest neighbors approach
└── README.md                            # Project documentation
```
---

**Contact**: [https://www.linkedin.com/in/danieldema/] | [danieldema42@gmail.com]

**Repository**: [https://github.com/danieldema/student_lifestyle]

**Data Source**: [Student Lifestyle Factors Dataset](https://www.kaggle.com/datasets/charlottebennett1234/lifestyle-factors-and-their-impact-on-students)
