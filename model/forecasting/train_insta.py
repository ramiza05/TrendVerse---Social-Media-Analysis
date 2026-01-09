import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
from flask import Flask, render_template_string

"""Importing Insta Trends Data"""

# insta trends

df_insta_trends = pd.read_csv(
    "C:/Users/M/Desktop/TrendVerse/model/forecasting/cleaned_insta_trends.csv",
    encoding='latin1'
)

content_types = [
    'media_type_Carousel','media_type_Photo','media_type_Reel','media_type_Video'
]

content_categories = [
    'content_category_Beauty','content_category_Comedy','content_category_Fashion',
    'content_category_Fitness','content_category_Food','content_category_Lifestyle',
    'content_category_Music','content_category_Photography','content_category_Technology',
    'content_category_Travel'
]

# Melt content type
types_long = df_insta_trends.melt(
    id_vars=['upload_date','reach'],
    value_vars=content_types,
    var_name='content_type',
    value_name='type_indicator'
)

types_long = types_long[types_long['type_indicator'] == 1]

# Melt content category
cats_long = df_insta_trends.melt(
    id_vars=['upload_date','reach'],
    value_vars=content_categories,
    var_name='content_category',
    value_name='cat_indicator'
)

cats_long = cats_long[cats_long['cat_indicator'] == 1]

# Merge type + category
merged = pd.merge(types_long[['upload_date','reach','content_type']],
                  cats_long[['upload_date','reach','content_category']],
                  on=['upload_date','reach'])

merged = merged.rename(columns={'upload_date':'ds','reach':'y'})
merged['ds'] = pd.to_datetime(merged['ds'])

forecasts = {}

for (ctype, cat), df_pair in merged.groupby(['content_type','content_category']):
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_pair[['ds','y']])

    future = model.make_future_dataframe(periods=365*2, freq='D')
    forecast = model.predict(future)

    forecasts[(ctype,cat)] = forecast[['ds','yhat']]

all_forecasts = []
for (ctype,cat), forecast in forecasts.items():
    f = forecast.copy()
    f['content_type'] = ctype
    f['content_category'] = cat
    f['ds'] = f['ds'].dt.normalize()
    all_forecasts.append(f)

combined = pd.concat(all_forecasts)

# For each date, pick the (type, category) with max yhat
trending_schedule = combined.loc[combined.groupby('ds')['yhat'].idxmax()]
trending_schedule = trending_schedule[['ds','content_category','yhat', 'content_type']].sort_values('ds')

# Filter for 2026
trending_2026 = trending_schedule[trending_schedule['ds'].dt.year == 2026]


import plotly.express as px
trending_schedule['label'] = trending_schedule['content_type'] + " + " + trending_schedule['content_category']
fig = px.line(trending_schedule, x='ds', y='yhat', color='label',
              title="Trending Content Type + Category Over Time")
# Convert to HTML fragment
graph_html = fig.to_html(full_html=False)

fig.write_html("artifacts/plot_html_2026.html", full_html=True, include_plotlyjs="cdn")
trending_schedule.to_csv("artifacts/trending_schedule_2026.csv", index=False)
#forecasts.to_csv("artifacts/forecasts_2025.csv", index=False)
