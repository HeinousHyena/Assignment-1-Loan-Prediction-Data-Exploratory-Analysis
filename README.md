Link to colab : https://colab.research.google.com/drive/1d2QFZp6dqFKnc9A6JVMLB3W41OKYPq3A#scrollTo=Ry3QDi7lGvuh

# Loan_Prediction_Data_Exp-loratory_Analysis
#**UNIVARIATE ANALYSIS**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

print (df.shape)
df.describe()

pd.crosstab(df['Credit_History'], df['Loan_Status'], margins=True )

# Plotting Data
df.boxplot('CoapplicantIncome')
df['CoapplicantIncome'].hist(bins=20)
df.boxplot('LoanAmount')
df['LoanAmount'].hist(bins=20)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df.boxplot(column='ApplicantIncome' , by='Education')

# Handling Missing values

print(df.isnull().sum())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Gender'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
print(df.isnull().sum())

df.boxplot('LoanAmount_log')

df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
df['Loan_Status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['red', 'grey'])

df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])

# Remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter data within the IQR bounds
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Remove outliers in 'ApplicantIncome' and 'LoanAmount'
df_no_outliers = remove_outliers_iqr(df, 'ApplicantIncome')
df_no_outliers = remove_outliers_iqr(df_no_outliers, 'LoanAmount')

# Check the resulting dataset
print(df_no_outliers.shape)

from scipy import stats

# Remove outliers using Z-Score
def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df[(np.abs(stats.zscore(df[columns])) < threshold).all(axis=1)]
    return df_clean

# Remove outliers in 'ApplicantIncome' and 'LoanAmount' using Z-Score
df_no_outliers_z = remove_outliers_zscore(df, ['ApplicantIncome', 'LoanAmount'])

# Check the resulting dataset
print(df_no_outliers_z.shape)





# **BIVARIATE ANALYSIS**

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (adjust the file path if needed)
# df = pd.read_csv('path_to_your_dataset.csv')

# List of categorical and numerical variables
categorical_vars = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
numerical_vars = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Set up the plot style
sns.set(style="whitegrid")

# Bivariate analysis for categorical variables against Loan_Status
for var in categorical_vars:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=var, hue='Loan_Status', data=df, palette='Set1')
    plt.title(f'Loan Status vs {var}')
    plt.show()

# Bivariate analysis for numerical variables against Loan_Status
for var in numerical_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Loan_Status', y=var, data=df, palette='Set2')
    plt.title(f'Loan Status vs {var}')
    plt.show()

# Correlation heatmap for numerical variables
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_vars].corr(), annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df, palette='coolwarm')
plt.title('Applicant Income vs Loan Amount')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.show()

# Scatter plot between CoapplicantIncome and LoanAmount
plt.figure(figsize=(8, 6))
sns.scatterplot(x='CoapplicantIncome', y='LoanAmount', hue='Loan_Status', data=df, palette='coolwarm')
plt.title('Coapplicant Income vs Loan Amount')
plt.xlabel('Coapplicant Income')
plt.ylabel('Loan Amount')
plt.show()

# Scatter plot between Credit_History and LoanAmount
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Credit_History', y='LoanAmount', hue='Loan_Status', data=df, palette='coolwarm')
plt.title('Credit History vs Loan Amount')
plt.xlabel('Credit History')
plt.ylabel('Loan Amount')
plt.show()

# Scatter plot between ApplicantIncome and CoapplicantIncome
plt.figure(figsize=(8, 6))
sns.scatterplot(x='ApplicantIncome', y='CoapplicantIncome', hue='Loan_Status', data=df, palette='coolwarm')
plt.title('Applicant Income vs Coapplicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Coapplicant Income')
plt.show()

import pandas as pd
import scipy.stats as stats
import numpy as np

# Load your dataset (adjust the file path if needed)
# df = pd.read_csv('path_to_your_dataset.csv')

# List of categorical and numerical variables
categorical_vars = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
numerical_vars = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# 1. Chi-Square Test for Categorical Variables

def chi_square_test(df, categorical_vars):
    print("Chi-Square Test Results:")
    for var in categorical_vars:
        contingency_table = pd.crosstab(df[var], df['Loan_Status'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"{var} vs Loan_Status:")
        print(f"Chi-Square = {chi2}, p-value = {p}")
        if p < 0.05:
            print(f"Significant association between {var} and Loan_Status\n")
        else:
            print(f"No significant association between {var} and Loan_Status\n")

# 2. T-Test for Numerical Variables

def t_test(df, numerical_vars):
    print("T-Test Results:")
    for var in numerical_vars:
        approved_loans = df[df['Loan_Status'] == 'Y'][var].dropna()  # Loans approved
        rejected_loans = df[df['Loan_Status'] == 'N'][var].dropna()  # Loans rejected
        t_stat, p_value = stats.ttest_ind(approved_loans, rejected_loans, equal_var=False)
        print(f"{var} vs Loan_Status:")
        print(f"T-statistic = {t_stat}, p-value = {p_value}")
        if p_value < 0.05:
            print(f"Significant difference in {var} between approved and rejected loans\n")
        else:
            print(f"No significant difference in {var} between approved and rejected loans\n")

# Running the tests
chi_square_test(df, categorical_vars)
t_test(df, numerical_vars)



