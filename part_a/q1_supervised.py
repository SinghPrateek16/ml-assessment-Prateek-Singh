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
...
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

          
     


