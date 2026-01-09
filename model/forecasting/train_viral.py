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
import plotly.express as px

"""Importing Viral Social Media Trends Data"""

# viral social media trends
df_viral_trends = pd.read_csv("C:/Users/M/Desktop/TrendVerse/model/forecasting/cleaned_viral_trends.csv", encoding="latin1",)

categories = [
    'Hashtag_#Music','Hashtag_#Fashion','Hashtag_#Education',
    'Hashtag_#Fitness','Hashtag_#Gaming','Hashtag_#Tech','Hashtag_#Viral'
]

views_long = df_viral_trends.melt(
    id_vars=['Post_Date','Views Cleaned'],
    value_vars=categories,
    var_name='category',
    value_name='indicator'
)

# Keep only rows where the hashtag is active
views_long = views_long[views_long['indicator'] == 1]

# ds, y
views_long = views_long.rename(columns={'Post_Date':'ds','Views Cleaned':'y'})
views_long['ds'] = pd.to_datetime(views_long['ds'])

forecasts = {}

for cat, df_cat in views_long.groupby('category'):
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_cat[['ds','y']])

    future = model.make_future_dataframe(periods=365, freq='D')
    forecast = model.predict(future)

    forecasts[cat] = forecast[['ds','yhat']]


all_forecasts = []
for cat, forecast in forecasts.items():
    f = forecast.copy()
    f['category'] = cat
    f['ds'] = f['ds'].dt.normalize()  # normalize dates
    all_forecasts.append(f)

combined = pd.concat(all_forecasts)

trending_schedule = combined.loc[combined.groupby('ds')['yhat'].idxmax()]
trending_schedule = trending_schedule[['ds','category','yhat']].sort_values('ds')


# trending_schedule has columns: ds, category, yhat
fig = px.line(trending_schedule, x='ds', y='yhat', color='category',
              title="Trending Categories Previously")

#logging to mlflow 

with mlflow.start.run():
    

graph_html = fig.to_html(full_html=False)

fig.write_html("artifacts/plot_html_prev.html", full_html=True, include_plotlyjs="cdn")
trending_schedule.to_csv("artifacts/trending_schedule_prev.csv", index=False)
#forecasts.to_csv("artifacts/forecasts_prev.csv", index=False)