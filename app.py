# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Set page config
st.set_page_config(page_title="Women's Safety Analysis Dashboard", layout="wide")

# File upload or local data loading
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ("Upload CSV", "Use Local Data")
)

@st.cache_data
def load_local_data():
    """Load data from local file"""
    try:
        df = pd.read_csv('most-dangerous-countries-for-women-2024.csv')
        return df
    except FileNotFoundError:
        st.error("Could not find the local data file. Please ensure 'most-dangerous-countries-for-women-2024.csv' is in the same directory as app.py")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def process_data(df):
    """Process and clean the dataframe"""
    if df is not None:
        # Clean column names
        df.columns = [col.replace('MostDangerousCountriesForWomen_', '') for col in df.columns]
        return df
    return None

# Data loading logic
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = process_data(df)
    else:
        st.warning("Please upload a CSV file to begin analysis.")
        st.stop()
else:
    df = load_local_data()
    df = process_data(df)
    if df is None:
        st.warning("Please ensure the data file is in the correct location or switch to file upload.")
        st.stop()

[... rest of the code remains the same as before ...]
