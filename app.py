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

# Set page config
st.set_page_config(page_title="Women's Safety Analysis Dashboard", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('most-dangerous-countries-for-women-2024.csv')
    # Clean column names
    df.columns = [col.replace('MostDangerousCountriesForWomen_', '') for col in df.columns]
    return df

df = load_data()

# Title and introduction
st.title("Global Women's Safety Analysis Dashboard")
st.write("""
This dashboard provides a comprehensive analysis of women's safety metrics across different countries.
The data includes various indicators such as the Women Peace and Security Index, street safety,
violence metrics, and legal discrimination measures.
""")

# Sidebar for filtering
st.sidebar.header("Filters")
selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Compare",
    options=['WomenPeaceAndSecurityIndex_Score_2023', 'WDIStreetSafety_2019', 
             'WDIIntentionalHomicide_2019', 'WDIIntimatePartnerViolence_2019'],
    default=['WomenPeaceAndSecurityIndex_Score_2023', 'WDIStreetSafety_2019']
)

# Main Analysis Sections
col1, col2 = st.columns(2)

with col1:
    st.header("Global Safety Score Distribution")
    fig = px.box(df, y='WomenPeaceAndSecurityIndex_Score_2023', 
                 title="Distribution of Women Peace and Security Index Scores")
    st.plotly_chart(fig)

with col2:
    st.header("Top 10 Safest Countries")
    top_10_safe = df.nlargest(10, 'WomenPeaceAndSecurityIndex_Score_2023')[
        ['country', 'WomenPeaceAndSecurityIndex_Score_2023']
    ]
    fig = px.bar(top_10_safe, x='country', y='WomenPeaceAndSecurityIndex_Score_2023',
                 title="Top 10 Countries by WPS Index")
    st.plotly_chart(fig)

# Correlation Analysis
st.header("Correlation Analysis")
correlation_metrics = [col for col in df.columns if col not in ['country'] and df[col].dtype in ['float64', 'int64']]
correlation_data = df[correlation_metrics].corr()

fig = px.imshow(correlation_data,
                labels=dict(color="Correlation Coefficient"),
                title="Correlation Matrix of Safety Metrics")
st.plotly_chart(fig)

# Regional Analysis
st.header("Regional Patterns")
# Create a scatter plot with multiple dimensions
fig = px.scatter(df, 
                 x='WomenPeaceAndSecurityIndex_Score_2023',
                 y='WDIStreetSafety_2019',
                 size='WDIIntimatePartnerViolence_2019',
                 color='WDIIntentionalHomicide_2019',
                 hover_data=['country'],
                 title="Multi-dimensional Safety Analysis")
st.plotly_chart(fig)

# Statistical Analysis
st.header("Statistical Analysis")
col3, col4 = st.columns(2)

with col3:
    # Perform K-means clustering
    features_for_clustering = ['WomenPeaceAndSecurityIndex_Score_2023', 'WDIStreetSafety_2019']
    clustering_data = df[features_for_clustering].dropna()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    clustering_df = clustering_data.copy()
    clustering_df['Cluster'] = clusters
    clustering_df['country'] = df['country'].iloc[clustering_data.index]
    
    fig = px.scatter(clustering_df,
                     x='WomenPeaceAndSecurityIndex_Score_2023',
                     y='WDIStreetSafety_2019',
                     color='Cluster',
                     hover_data=['country'],
                     title="Country Clusters Based on Safety Metrics")
    st.plotly_chart(fig)

with col4:
    # Calculate and display summary statistics
    st.subheader("Summary Statistics")
    summary_stats = df[['WomenPeaceAndSecurityIndex_Score_2023', 'WDIStreetSafety_2019']].describe()
    st.write(summary_stats)

# Time Series Analysis (if temporal data is available)
st.header("Violence Metrics Comparison")
violence_metrics = ['WDIStreetSafety_2019', 'WDIIntentionalHomicide_2019', 
                   'WDINonPartnerViolence_2019', 'WDIIntimatePartnerViolence_2019']
violence_data = df.melt(id_vars=['country'], 
                       value_vars=violence_metrics,
                       var_name='Metric', 
                       value_name='Score')

fig = px.box(violence_data, x='Metric', y='Score',
             title="Distribution of Violence Metrics Across Countries")
st.plotly_chart(fig)

# Risk Assessment
st.header("Risk Assessment")
# Create a composite risk score
risk_metrics = ['WDIStreetSafety_2019', 'WDIIntentionalHomicide_2019', 
                'WDINonPartnerViolence_2019', 'WDIIntimatePartnerViolence_2019']

df['RiskScore'] = df[risk_metrics].mean(axis=1)
top_risk = df.nlargest(10, 'RiskScore')[['country', 'RiskScore']]

fig = px.bar(top_risk, x='country', y='RiskScore',
             title="Top 10 Countries by Composite Risk Score")
st.plotly_chart(fig)

# Recommendations and Insights
st.header("Key Insights")
st.write("""
### Main Findings:
1. There is significant variation in women's safety across countries
2. Strong correlation between street safety and overall security index
3. Distinct regional patterns in safety metrics
4. Some countries show high scores in certain areas but lag in others

### Recommendations:
1. Focus on improving street safety in high-risk areas
2. Strengthen legal frameworks in countries with high discrimination scores
3. Implement targeted interventions based on cluster analysis
4. Develop comprehensive safety programs addressing multiple risk factors
""")

# Download section
st.header("Download Data")
st.download_button(
    label="Download analyzed data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='women_safety_analysis.csv',
    mime='text/csv',
)

# Footer
st.markdown("---")
st.markdown("Dashboard created using Streamlit and Python | Data source: Women's Safety Index 2024")
