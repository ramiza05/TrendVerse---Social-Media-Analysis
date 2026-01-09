import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats as stats
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

"""Importing Viral Social Media Trends Data"""

# viral social media trends

df_viral_trends = pd.read_csv("C:/Users/M/Desktop/TrendVerse/data/raw/Cleaned_Viral_Social_Media_Trends.csv",
    encoding="latin1",
)


"""EDA - checking for missing values/outliers and encoding"""

# outliers in df_viral_trends:

columns = ["Views","Likes","Shares","Comments"	]

for col in columns:
  new = col + " Cleaned"
  df_viral_trends[new] = stats.zscore(df_viral_trends[col])
  df_viral_trends_cleaned = df_viral_trends[(df_viral_trends[new]>-3) & (df_viral_trends[new]<3)].reset_index()

df_viral_trends_cleaned = df_viral_trends_cleaned.drop(columns=["Views","Likes","Shares","Comments", "index", "Post_ID"], axis = 1)


# encoding platform, hashtag, content type, region, engagement level

columns = ["Platform","Hashtag","Content_Type","Region","Engagement_Level"]

encoder = OneHotEncoder( sparse_output=False)
encoded_columns = encoder.fit_transform(df_viral_trends_cleaned[columns])

df_viral = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out())

df_viral.columns

columns = ["Platform","Hashtag","Content_Type","Region","Engagement_Level"]
df_viral_trends_cleaned.drop(columns=columns, axis=1)

df_viral_trends_encoded = pd.concat([df_viral_trends_cleaned, df_viral], axis=1)

df_viral_trends_encoded.columns

temp_df = df_viral_trends_encoded['Engagement_Level_High']
temp_df = pd.concat([temp_df, df_viral_trends_encoded['Engagement_Level_Medium']], axis=1)
temp_df = pd.concat([temp_df, df_viral_trends_encoded['Engagement_Level_Low']], axis=1)
temp_df["Engagement"] = temp_df.idxmax(axis=1)
temp_df.head()
temp_df["Engagement_Cleaned"]=temp_df["Engagement"].map({
    'Engagement_Level_Low':1,
    'Engagement_Level_Medium':2,
    'Engagement_Level_High':3
})

"""Converting Date to Integer"""

# convert to datetime
df_viral_trends_encoded['Post_Date'] = pd.to_datetime(df_viral_trends_encoded['Post_Date'], format='%Y-%m-%d', errors='coerce')
df_viral_trends_encoded.head()

# get the minimum date

earliest_date = df_viral_trends_encoded['Post_Date'].min
df_viral_trends_encoded = df_viral_trends_encoded.sort_values('Post_Date')

# extracting numeric features - engagements (lagged, normalized), recency, and seasonality
# my engagement is categorical - prophet needs a numeric series to work with

# get the rows' column that is the max for all engagement
df_viral_trends_encoded["y"] = temp_df["Engagement_Cleaned"]

# lagged and normalized engagement

df_viral_trends_encoded["Engagement_Lag1"]=df_viral_trends_encoded["y"].shift(1)
df_viral_trends_encoded["Engagement_Lag7"]=df_viral_trends_encoded["y"].shift(7)

# using simple imputer to delete missing values
imp = SimpleImputer(strategy='median')
df_viral_trends_encoded[["Engagement_Lag1"]]=imp.fit_transform(df_viral_trends_encoded[["Engagement_Lag1"]])
df_viral_trends_encoded[["Engagement_Lag7"]]=imp.fit_transform(df_viral_trends_encoded[["Engagement_Lag7"]])

# recency
df_viral_trends_encoded["recency_days"] = (df_viral_trends_encoded['Post_Date'].max()-df_viral_trends_encoded['Post_Date']).dt.days
df_viral_trends_encoded["recency_days_norm"]=df_viral_trends_encoded['recency_days']/df_viral_trends_encoded['recency_days'].max()

# seasonality
df_viral_trends_encoded['dow'] = df_viral_trends_encoded['Post_Date'].dt.dayofweek   # 0=Monday
df_viral_trends_encoded['month'] = df_viral_trends_encoded['Post_Date'].dt.month

df_viral_trends_encoded['Engagement_Lag1']

df_viral_trends_encoded.to_csv("processed/cleaned_viral_trends.csv", index=False)