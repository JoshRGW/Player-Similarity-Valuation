#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import re


# # 2. Data Preprocessing

# In[2]:


# Load the transfermarkt dataset
tm_df = pd.read_csv('transfermarkt.csv', low_memory=False)


# In[3]:


tm_df.head()


# In[4]:


tm_df.info()


# In[5]:


tm_df.describe()


# In[6]:


# Plotting Distribution of Player Market Values

# Setting up the plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plotting histograms
sns.histplot(tm_df['valueeur'], bins=30, kde=False, ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Value in EUR')
axes[0, 0].set_xlabel('Value in EUR')
axes[0, 0].set_ylabel('Frequency')

sns.histplot(tm_df['wageeur'], bins=30, kde=False, ax=axes[0, 1])
axes[0, 1].set_title('Histogram of Wage in EUR')
axes[0, 1].set_xlabel('Wage in EUR')
axes[0, 1].set_ylabel('Frequency')

# Plotting density plots
sns.kdeplot(tm_df['valueeur'], ax=axes[1, 0])
axes[1, 0].set_title('Density Plot of Value in EUR')
axes[1, 0].set_xlabel('Value in EUR')
axes[1, 0].set_ylabel('Density')

sns.kdeplot(tm_df['wageeur'], ax=axes[1, 1])
axes[1, 1].set_title('Density Plot of Wage in EUR')
axes[1, 1].set_xlabel('Wage in EUR')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.show()


# In[7]:


# Checking for missing values in the relevant columns
columns_of_interest = ['valueeur', 'wageeur', 'age', 'clubjoineddate', 'clubcontractvaliduntilyear', 'releaseclauseeur']
missing_values = tm_df[columns_of_interest].isnull().sum()
print("\nMissing values before imputation:")
print(missing_values)


# In[8]:


# Impute missing values
tm_df['valueeur'].fillna(tm_df['valueeur'].median(), inplace=True)
tm_df['wageeur'].fillna(tm_df['wageeur'].median(), inplace=True)
tm_df['clubjoineddate'].fillna('2018-01-01', inplace=True)
tm_df['clubcontractvaliduntilyear'].fillna(0, inplace=True)
tm_df['releaseclauseeur'].fillna(tm_df['releaseclauseeur'].median(), inplace=True)


# In[9]:


# Check if missing values are handled
missing_values_after = tm_df[columns_of_interest].isnull().sum()
print("\nMissing values after imputation:")
print(missing_values_after)


# # 3. Random Forest Regressor Model

# In[10]:


# Feature selection for the regression model
features = ['age', 'wageeur', 'clubcontractvaliduntilyear', 'releaseclauseeur']
X = tm_df[features]
y = tm_df['valueeur']


# In[11]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[13]:


# Predictions
y_pred = model.predict(X_test)


# # 4. Model Evaluation

# In[14]:


# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel evaluation:")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")


# In[15]:


# Predict player valuations using the trained model
tm_df['estimated_value_eur'] = model.predict(tm_df[features])


# In[16]:


# Display the first few rows of the dataset with predicted valuations
print("\nFirst 10 rows of the dataset with predicted valuations:")
print(tm_df[['longname', 'valueeur', 'estimated_value_eur']].head(10))


# # 5. Model Visualisation

# In[17]:


# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


# In[18]:


# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()


# In[19]:


# Predictions vs. Actual Values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
plt.show()


# In[20]:


# Save the dataset with predicted valuations to a CSV file
tm_df.to_csv('player_valuations.csv', index=False)
print("\nData with predicted valuations saved to 'player_valuations.csv'")


# In[ ]:




