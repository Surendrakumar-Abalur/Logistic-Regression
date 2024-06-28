#!/usr/bin/env python
# coding: utf-8

# # 1. Data Exploration:

# In[1]:


# Load the training dataset into a DataFrame
import pandas as pd
Training_data=pd.read_csv("E:/DS ASSIGNMENTS/Logistic Regression/Titanic_train.csv")
Training_data.head()


# In[2]:


# Load the testing dataset into a DataFrame
Testing_data=pd.read_csv("E:/DS ASSIGNMENTS/Logistic Regression/Titanic_test.csv")
Testing_data.tail()


# In[3]:


# Concatinating training and testing data 
Titanic_Data = pd.concat([Training_data, Testing_data], ignore_index=True, sort  = False)
Titanic_Data.tail()


# In[4]:


# number of rows and columns in dataset in python
Titanic_Data.shape


# In[5]:


# View the first few rows of the data frame.
Titanic_Data.head()


# In[6]:


# Data Information
Titanic_Data.info()


# In[7]:


# Data Description
Titanic_Data.describe()


# In[8]:


# Get the data types of each column.
Titanic_Data.dtypes


# In[9]:


# visualizations such as histograms, box plots, or pair plot 
import matplotlib.pyplot as plt 
import seaborn as sns

# Create a histogram of the 'Age' column.
plt.hist(Titanic_Data['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()

# Create a box plot of the 'Fare' column.
sns.boxplot(x=Titanic_Data['Fare'])
plt.xlabel('Fare')
plt.ylabel('Amount')
plt.title('Distribution of Fare')
plt.show()

# Create a pair plot of the numerical columns.
sns.pairplot(Titanic_Data[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()


# In[10]:


# Analysed and observed patterns or correlations  in the data.

# Analyze the relationship between survival and passenger class.
sns.barplot(x='Pclass', y='Survived', data=Titanic_Data)
plt.show()

# Analyze the relationship between survival and sex.
sns.barplot(x='Sex', y='Survived', data=Titanic_Data)
plt.show()

# Analyze the relationship between survival and age.
sns.histplot(x='Age', hue='Survived', data=Titanic_Data)
plt.show()

# Analyze the relationship between survival and fare.
sns.histplot(x='Fare', hue='Survived', data=Titanic_Data)
plt.show()


# Analyze the correlation between numerical columns.
correlation_matrix = Titanic_Data[['Survived', 'Pclass', 'Age', 'Fare']].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Analyze the correlation between numerical columns.
print(Titanic_Data[['Survived', 'Pclass', 'Age', 'Fare']].corr())


# # 2. Data Preprocessing:

# In[11]:


# Identify the columns with missing values and their Percentage
missing_values = Titanic_Data.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / len(Titanic_Data)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
missing_data


# In[12]:


# prompt: Handle missing values (e.g., imputation).
# Drop rows with NaN values in y
Titanic_Data.dropna(subset=['Survived'], inplace=True)
# Impute missing values Titanic_Data'Cabin'
Titanic_Data['Cabin'] = Titanic_Data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
# Impute missing values in the 'Age' column with the mean age.
Titanic_Data['Age'].fillna(Titanic_Data['Age'].mean(), inplace=True)
# Impute missing values in the 'Embarked' column with the mode 
Titanic_Data.Embarked.fillna(Titanic_Data.Embarked.mode()[0], inplace = True)
# Impute missing values in the 'Fare' column with the median fare.
Titanic_Data['Fare'].fillna(Titanic_Data['Fare'].median(), inplace=True)
# Check for remaining missing values.
Titanic_Data.isnull().sum()


# In[13]:


#Find categorical variables
data_types = Titanic_Data.dtypes

# Initialize an empty list to store categorical columns
categorical_columns = []

# Iterate through the columns and find categorical columns
for column, dtype in data_types.items():
    if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
        categorical_columns.append(column)

# Print the categorical columns found
print("Categorical columns:")
print(categorical_columns)


# In[14]:


# Assuming df is your DataFrame after preprocessing
DT = Titanic_Data.drop(columns=['PassengerId', 'Name','Cabin','Ticket', 'Survived'])
DT.head()


# In[15]:


#Encode categorical variables using one-hot encoding for Embarked
df_encoded = pd.get_dummies(DT, columns=['Embarked'])
#Encode categorical variables using LabelEncoder for
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
#LE.fit_transformation
DT["Sex"] = LE.fit_transform(DT["Sex"])
X_contineuous = df_encoded[['Pclass','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']]
X_contineuous


# In[16]:


# Tranform the data into StandardScalr
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X_contineuous)
SS_X = pd.DataFrame(SS_X) 
d1 = ['Pclass','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']
SS_X.columns = d1
SS_X.head()


# # 3. Model Building:

# In[17]:


# Drop rows with NaN values in y
Titanic_Data.dropna(subset=['Survived'], inplace=True)
# Data partition
X = pd.concat([SS_X,DT["Sex"]], axis=1)
Y = Titanic_Data['Survived']


# In[18]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test   = train_test_split(X,Y,test_size=0.32,random_state=42)


# In[19]:


# Train the model using the training data.
from sklearn.linear_model import LogisticRegression
Logreg = LogisticRegression()
Logreg.fit(X_test,Y_test)
y_pred_test = Logreg.predict(X_test)
y_pred_test


# # 4. Model Evaluation:

# In[20]:


# prediction of confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,y_pred_test)


# In[21]:


# prediction of Accuracy Score
AS = accuracy_score(Y_test,y_pred_test)
print("Accuracy score:", AS.round(3))


# In[22]:


# predictions of Snsitivity,Precision and F1 score of test data
from sklearn.metrics import recall_score, precision_score,f1_score
print("Sensitivity score: ", recall_score(Y_test,y_pred_test).round(2))
print("Precision score: ", precision_score(Y_test,y_pred_test).round(2))
print("F1 score: ", f1_score(Y_test,y_pred_test).round(2))


# In[23]:


# predictions of Specificity
cm = confusion_matrix(Y_test,y_pred_test)
TN = cm[0,0]
FP = cm[0,1]
TNR = TN/(TN + FP)
print("specificity score: ", TNR.round(2))


# In[24]:


# printing the original probabilities and saving under data

Titanic_Data["y_proba"] = Logreg.predict_proba(X)[:,1]  # 1-prob, prob
Titanic_Data.head()


# In[25]:


# predictions of ROC-AUC score.
from sklearn.metrics import roc_curve,roc_auc_score
# fpr = 1 - spcificity , tpr = sensitivity
fpr,tpr,dummy = roc_curve(Titanic_Data["Survived"],Titanic_Data["y_proba"])


# In[26]:


# Visualize the ROC curve.
import matplotlib.pyplot as plt
plt.scatter(fpr, tpr)
plt.plot(fpr, tpr,color='red')
plt.xlabel("1- specificity")
plt.ylabel("sensitivity")
plt.show()
print("Area under curve:", roc_auc_score(Titanic_Data["Survived"],Titanic_Data["y_proba"]))


# # 5. Interpretation:

# In[28]:


# Interpret the coefficients of the logistic regression model.
# Access the coefficients
coefficients = Logreg.coef_

# Create a DataFrame with the feature names and coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients[0]})

# Sort the DataFrame by the absolute value of the coefficients
coef_df = coef_df.sort_values(by='Coefficient', key=abs)

# Print the DataFrame
print(coef_df)

# Interpret the coefficients
for i, row in coef_df.iterrows():
    feature_name = row['Feature']
    coefficient = row['Coefficient']

    print(f"**Feature:** {feature_name}")
    print(f"**Coefficient:** {coefficient:.3f}")

    if coefficient > 0:
        print(f"{feature_name} is positively associated with survival.")
    else:
        print(f"{feature_name} is negatively associated with survival.")


# In[ ]:


# Discuss the significance of features in predicting the target variable (survival probability in this case).
# Based on the coefficients, we can see that the most significant features in predicting survival are:
# - Sex: Being female is positively associated with survival.
# - Pclass: Higher class passengers are more likely to survive.
# - Age: Younger passengers are more likely to survive.
# - Fare: Passengers who paid a higher fare are more likely to survive.
# - Embarked_C: Passengers who embarked from Cherbourg are more likely to survive.
# - Embarked_Q: Passengers who embarked from Queenstown are less likely to survive.
# - Embarked_S: Passengers who embarked from Southampton are less likely to survive.
# - SibSp: Passengers with more siblings or spouses aboard are less likely to survive.
# - Parch: Passengers with more parents or children aboard are less likely to survive.

