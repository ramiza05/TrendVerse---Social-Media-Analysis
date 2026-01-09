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

df_insta_analytics = pd.read_csv("C:/Users/M/Desktop/TrendVerse/data/raw/Instagram_Analytics.csv", encoding='latin1')

df_insta_analytics['upload_date']=pd.to_datetime(df_insta_analytics['upload_date'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
df_insta_analytics.head()

# get the minimum date

earliest_d = df_insta_analytics['upload_date'].min
df_insta_analytics = df_insta_analytics.sort_values('upload_date')

df_insta_analytics=df_insta_analytics.drop('post_id', axis=1)
# encoding platform, hashtag, content type, region, engagement level

columns = ["media_type" , "content_category", "traffic_source"]

encoder = OneHotEncoder( sparse_output=False)
encoded_columns = encoder.fit_transform(df_insta_analytics[columns])

df_insta_a = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out())
df_analytics = pd.concat([df_insta_analytics, df_insta_a], axis=1)
df_analytics=df_analytics.drop(columns=["media_type" , "content_category", "traffic_source"], axis=1)

df_analytics.to_csv("processed/cleaned_insta_trends.csv", index=False)