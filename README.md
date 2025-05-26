# Student Lifestyle & Academic Performance Analysis

A comprehensive machine learning study examining the relationship between student lifestyle factors and academic performance, featuring comparative analysis across four regression algorithms with hyperparameter optimization.

## Project Overview

This project investigates how various lifestyle factors impact student academic performance through predictive modeling. By analyzing sleep patterns, study habits, social activities, and stress levels, the study provides insights into optimal lifestyle choices for academic success. The project implements and compares four different machine learning approaches to understand the underlying relationships between lifestyle and grades.

### Key Achievements
- **Multi-Algorithm Comparison** - Comprehensive evaluation of 4 regression models
- **Hyperparameter Optimization** - GridSearchCV implementation for optimal performance
- **Feature Importance Analysis** - Identification of key lifestyle predictors
- **Statistical Insights** - Linear relationship discovery through model comparison

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
- **Interpretability**: Clear decision rules for grade prediction
- **Overfitting Prevention**: Pruning through depth limitation

### Random Forest Implementation (student_lifestyle_randomforest.ipynb)
- **Ensemble Method**: Multiple decision trees with aggregation
- **Configuration**: max_depth = 3 for consistency
- **Performance**: R² = 0.51, MSE = 0.25
- **Robustness**: Reduced variance through ensemble averaging
- **Feature Importance**: Comprehensive ranking of lifestyle factors

### K-Nearest Neighbors Modeling (student_lifestyle_knn.ipynb)
- **Optimization**: GridSearchCV for neighbor count
- **Best Parameters**: n_neighbors = 28
- **Performance**: R² = 0.50, MSE = 0.26
- **Non-parametric Approach**: Instance-based learning
- **Local Patterns**: Similarity-based predictions

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

## Technical Methodology

### Model Development Process
1. **Exploratory Data Analysis**: Initial pattern identification
2. **Feature Engineering**: Variable transformation and selection
3. **Model Implementation**: Algorithm-specific development
4. **Hyperparameter Tuning**: GridSearchCV optimization
5. **Performance Evaluation**: Comprehensive metric analysis
6. **Cross-Validation**: Robust performance estimation

### Evaluation Metrics
- **R-squared Score**: Explained variance measurement
- **Mean Squared Error**: Prediction accuracy assessment
- **Cross-Validation**: Generalization capability testing

## Project Structure

```
├── student_lifestyle_linreg.ipynb       # Linear regression implementation
├── student_lifestyle_decisiontree.ipynb # Decision tree analysis
├── student_lifestyle_randomforest.ipynb # Random forest modeling
├── student_lifestyle_knn.ipynb          # K-nearest neighbors approach
└── README.md                            # Project documentation
```

## Business & Educational Implications

### Academic Counseling Insights
- **Study Optimization**: Evidence-based study hour recommendations
- **Lifestyle Balance**: Holistic approach to student well-being
- **Stress Management**: Targeted interventions for high-stress students
- **Performance Prediction**: Early warning systems for academic risk

### Policy Recommendations
- **Sleep Education**: Importance of adequate rest for academic success
- **Activity Programming**: Balanced extracurricular and physical activity
- **Support Systems**: Stress reduction and mental health resources
- **Personalized Approaches**: Individual lifestyle optimization strategies

## Technical Skills Demonstrated

- **Machine Learning**: Multi-algorithm implementation and comparison
- **Hyperparameter Optimization**: GridSearchCV and model tuning
- **Statistical Analysis**: Performance evaluation and interpretation
- **Python Programming**: Advanced scikit-learn library usage
- **Model Evaluation**: Comprehensive performance assessment

## Research Limitations & Considerations

### Data Constraints
- **Feature Limitation**: Current variables explain ~52% of variance
- **Sample Representation**: Dataset scope and generalizability
- **Causality**: Correlation vs. causal relationships

### Model Considerations
- **Linear Assumptions**: Predominant linear relationships observed
- **Complexity Trade-offs**: Simple vs. complex model performance
- **Overfitting Risk**: Model generalization capabilities
- **Feature Interactions**: Potential unmeasured variable effects

---

**Contact**: [https://www.linkedin.com/in/danieldema/] | [danieldema42@gmail.com]

**Repository**: [https://github.com/danieldema/student_lifestyle]

**Data Source**: [Student Lifestyle Factors Dataset](https://www.kaggle.com/datasets/charlottebennett1234/lifestyle-factors-and-their-impact-on-students)
