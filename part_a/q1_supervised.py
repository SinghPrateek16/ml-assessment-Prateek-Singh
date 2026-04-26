

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# LOAD DATASET (upload file manually in Colab)
data = pd.read_csv('q1_heart_disease.csv')

print("Dataset loaded successfully!")



# BASIC DATA INSPECTION

print("\nShape of dataset:", data.shape)

print("\nData types:\n", data.dtypes)

print("\nMissing values:\n", data.isnull().sum())

print("\nFirst 5 rows:")
print(data.head())
     
Dataset loaded successfully!

Shape of dataset: (800, 12)

Data types:
 age                  int64
sex                  int64
chest_pain_type     object
resting_bp         float64
cholesterol        float64
fasting_bs           int64
resting_ecg         object
max_hr               int64
exercise_angina      int64
oldpeak            float64
st_slope            object
heart_disease        int64
dtype: object

Missing values:
 age                 0
sex                 0
chest_pain_type     0
resting_bp         24
cholesterol        32
fasting_bs          0
resting_ecg         0
max_hr              0
exercise_angina     0
oldpeak             0
st_slope            0
heart_disease       0
dtype: int64

First 5 rows:
   age  sex  chest_pain_type  resting_bp  cholesterol  fasting_bs  \
0   68    0  atypical_angina       142.0        399.0           0   
1   58    1      non_anginal       163.0        310.0           1   
2   44    1      non_anginal       128.0        175.0           0   
3   72    1     asymptomatic       114.0        177.0           0   
4   37    1      non_anginal       149.0        271.0           0   

                    resting_ecg  max_hr  exercise_angina  oldpeak st_slope  \
0  left_ventricular_hypertrophy     169                0      0.4       up   
1         st_t_wave_abnormality     121                1      1.1       up   
2                        normal     183                1      0.2       up   
3         st_t_wave_abnormality     150                0      1.0       up   
4                        normal     136                0      0.4     flat   

   heart_disease  
0              1  
1              1  
2              0  
3              1  
4              0  

# EXPLORATORY DATA ANALYSIS (EDA)

# 1. Target distribution
sns.countplot(x='heart_disease', data=data)
plt.title("Distribution of Heart Disease")
plt.show()

# 2. Correlation heatmap (FIXED VERSION)
plt.figure(figsize=(10,8))

# select only numeric columns
numeric_data = data.select_dtypes(include=np.number)

sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# 3. Age vs Heart Disease
sns.boxplot(x='heart_disease', y='age', data=data)
plt.title("Age vs Heart Disease")
plt.show()


     



Exploratory Data Analysis (EDA)
From the visualizations:

The target distribution plot shows how many patients have heart disease versus those who do not. The classes appear fairly balanced, which is good because the model will not be biased toward one class.

The correlation heatmap shows relationships between numerical features. Some variables show moderate correlation with each other and with the target variable, suggesting they may be useful for predicting heart disease.

The age vs heart disease boxplot indicates that patients with heart disease tend to be slightly older on average compared to those without the disease.

Overall, these visualizations help in understanding the data patterns and confirm that the dataset is suitable for building a classification model.


# DATA PREPROCESSING


# Handle missing values with a documented strategy
data.fillna(data.median(numeric_only=True), inplace=True)

# Convert categorical variables into numbers
data = pd.get_dummies(data, drop_first=True)

# Split into features (X) and target (y)
X = data.drop('heart_disease', axis=1)
y = data['heart_disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


     

# MODEL TRAINING

decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
gradient_boosting_model.fit(X_train, y_train)
     
GradientBoostingClassifier(random_state=42)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# MODEL EVALUATION


models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Gradient Boosting": gradient_boosting_model
}

for name, model in models.items():
    print("\n")
    print("Model:", name)

    predictions = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))



# HYPERPARAMETER TUNING (Random Forest)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)



# EVALUATE TUNED MODEL


best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

print("\n")
print("Tuned Random Forest Performance")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_predictions))

print("\nClassification Report:")
print(classification_report(y_test, tuned_predictions))
     

Model: Decision Tree

Confusion Matrix:
[[57 22]
 [25 56]]

Classification Report:
              precision    recall  f1-score   support

           0       0.70      0.72      0.71        79
           1       0.72      0.69      0.70        81

    accuracy                           0.71       160
   macro avg       0.71      0.71      0.71       160
weighted avg       0.71      0.71      0.71       160



Model: Random Forest

Confusion Matrix:
[[61 18]
 [15 66]]

Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.77      0.79        79
           1       0.79      0.81      0.80        81

    accuracy                           0.79       160
   macro avg       0.79      0.79      0.79       160
weighted avg       0.79      0.79      0.79       160



Model: Gradient Boosting

Confusion Matrix:
[[61 18]
 [19 62]]

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.77      0.77        79
           1       0.78      0.77      0.77        81

    accuracy                           0.77       160
   macro avg       0.77      0.77      0.77       160
weighted avg       0.77      0.77      0.77       160


Best parameters: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}


Tuned Random Forest Performance

Confusion Matrix:
[[57 22]
 [15 66]]

Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.72      0.75        79
           1       0.75      0.81      0.78        81

    accuracy                           0.77       160
   macro avg       0.77      0.77      0.77       160
weighted avg       0.77      0.77      0.77       160

Model Comparison and Best Model
After testing all three models — Decision Tree, Random Forest and Gradient Boosting — we can compare how well they perform using precision, recall and F1-score.

The Decision Tree model works okay, but it seems a bit unstable and doesn’t perform as well as the others. Gradient Boosting improves the results, but it is still slightly behind Random Forest.

Random Forest gives the best overall performance. It has a higher F1-score compared to the other models, which means it balances precision and recall better. This is important in this case because we want to correctly identify patients with heart disease while also avoiding too many wrong predictions.

So instead of just looking at accuracy, using F1-score helps us understand the model performance more clearly.

Final Conclusion:
Random Forest is the best model for this problem because it provides the most balanced and reliable results across precision, recall and F1-score.



# COMPARE BASELINE vs TUNED MODEL


# Baseline Random Forest (before tuning)
print("\nBaseline Random Forest Performance:")
baseline_pred = random_forest_model.predict(X_test)
print(classification_report(y_test, baseline_pred))


# Tuned Random Forest (after tuning)
print("\nTuned Random Forest Performance:")
tuned_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, tuned_pred))


# Best parameters
print("\nBest Parameters Found:")
print(grid_search.best_params_)
     
Baseline Random Forest Performance:
              precision    recall  f1-score   support

           0       0.80      0.77      0.79        79
           1       0.79      0.81      0.80        81

    accuracy                           0.79       160
   macro avg       0.79      0.79      0.79       160
weighted avg       0.79      0.79      0.79       160


Tuned Random Forest Performance:
              precision    recall  f1-score   support

           0       0.79      0.72      0.75        79
           1       0.75      0.81      0.78        81

    accuracy                           0.77       160
   macro avg       0.77      0.77      0.77       160
weighted avg       0.77      0.77      0.77       160


Best Parameters Found:
{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
Hyperparameter Tuning
I used GridSearchCV to tune the Random Forest model by trying different values for parameters like the number of trees, tree depth and minimum samples required to split.

The best parameters I got were: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}

Comparison of Models
After comparing the results of the baseline and tuned models, I noticed that the baseline model actually performs slightly better.

The baseline model has an accuracy of 0.79, while the tuned model has 0.77.
The average F1-score for the baseline model is 0.79, compared to 0.77 for the tuned model.
For predicting patients with heart disease (class 1), the baseline model has an F1-score of 0.80, while the tuned model has 0.78.
So overall, the tuned model did not improve the performance and is slightly worse than the original model.

Conclusion
In this case, hyperparameter tuning did not help improve the model. The original Random Forest model still gives better and more reliable results, so it is the better choice for this problem.

