#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd



# Load the dataset
data_path = "E:/DS ASSIGNMENTS/Logistic Regression/Titanic_train.csv"
df = pd.read_csv(data_path)

# Title and description
st.title("Titanic Data Analysis")
st.markdown("This app allows you to explore the Titanic dataset.")

# Show the dataset
st.subheader("Dataset")
st.write(df.head())

# Sidebar filters
st.sidebar.header("Filters")
pclass = st.sidebar.multiselect("Passenger Class", options=df['Pclass'].unique(), default=df['Pclass'].unique())
sex = st.sidebar.multiselect("Sex", options=df['Sex'].unique(), default=df['Sex'].unique())
embarked = st.sidebar.multiselect("Port of Embarkation", options=df['Embarked'].unique(), default=df['Embarked'].unique())

# Filter the data
filtered_df = df[(df['Pclass'].isin(pclass)) & (df['Sex'].isin(sex)) & (df['Embarked'].isin(embarked))]

# Show filtered data
st.subheader("Filtered Dataset")
st.write(filtered_df)



