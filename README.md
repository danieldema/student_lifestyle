The data used in this project was obtained from Kaggle: https://www.kaggle.com/datasets/charlottebennett1234/lifestyle-factors-and-their-impact-on-students

In student_lifestyle_linreg.ipynb, I use scikit-learn to build a linear regression model to predict student grades based on lifestyle factors including study hours per day, extracurricular hours per day, sleep hours per day, social hours per day, physical activity hours per day, stress level, and gender. This model predicts grades with R-squared score approximately 0.52 and MSE approximately 0.25.

In student_lifestyle_decisiontree.ipynb, I use the same features to build a decision tree to predict grades. GridSearchCV gives an optimal tree depth of 3, and this model predicts grades with R-squared score approximately 0.51 and MSE approximately 0.26.

In student_lifestyle_randomforest.ipynb, I use the same features to build a random forest to predict grades, again with tree depth of 3. This model predicts grades with R-squared score approximately 0.51 and MSE approximately 0.25.

In student_lifestyle_knn, I use the same features to build a KNN regressor to predict grades. GridSearchCV gives an optimal number of 28 neighbors, and this model predicts grades with R-squared score approximately 0.5 and MSE approximately 0.26.

The similarity in R-squared scores and MSE of the last three models to those of the linear regression model suggests that the relationship between the features and grades is approximately linear, and is indicative of possible limitations in the data.
