import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configure Streamlit for better performance
st.set_page_config(
    page_title="Women's Safety Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache data loading
@st.cache_data
def load_local_data():
    """Load and preprocess data"""
    try:
        df = pd.read_csv('most-dangerous-countries-for-women-2024.csv')
        # Clean column names once during load
        df.columns = [col.replace('MostDangerousCountriesForWomen_', '') for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Cache statistical calculations
@st.cache_data
def calculate_statistics(df):
    """Calculate all statistical measures at once"""
    if df is None:
        return None
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {
        'correlation_matrix': df[numeric_cols].corr(),
        'summary_stats': df[numeric_cols].describe(),
        'top_10_safe': df.nlargest(10, 'WomenPeaceAndSecurityIndex_Score_2023'),
        'risk_metrics': ['WDIStreetSafety_2019', 'WDIIntentionalHomicide_2019', 
                        'WDINonPartnerViolence_2019', 'WDIIntimatePartnerViolence_2019']
    }
    return stats

# Cache plotting data preparation
@st.cache_data
def prepare_plot_data(df, stats):
    """Prepare all data needed for plotting"""
    if df is None or stats is None:
        return None
    
    plot_data = {
        'scatter_data': df[['WomenPeaceAndSecurityIndex_Score_2023', 'WDIStreetSafety_2019']].dropna(),
        'violence_data': df.melt(
            id_vars=['country'], 
            value_vars=stats['risk_metrics'],
            var_name='Metric', 
            value_name='Score'
        ).dropna()
    }
    return plot_data

# Load data once at startup
df = load_local_data()

if df is not None:
    # Calculate all statistics at once
    stats = calculate_statistics(df)
    # Prepare all plot data at once
    plot_data = prepare_plot_data(df, stats)

    # Title and introduction
    st.title("Global Women's Safety Analysis Dashboard")
    st.write("""
    This dashboard analyzes women's safety metrics across different countries,
    including the Women Peace and Security Index, street safety, violence metrics,
    and legal discrimination measures.
    """)

    # Sidebar filters
    st.sidebar.header("Filters")
    available_metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Compare",
        options=available_metrics,
        default=available_metrics[:2]
    )

    # Main dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Safety Score Distribution")
        fig1 = px.box(df, y='WomenPeaceAndSecurityIndex_Score_2023',
                     title="Distribution of Women Peace and Security Index Scores")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.header("Top 10 Safest Countries")
        fig2 = px.bar(stats['top_10_safe'],
                     x='country',
                     y='WomenPeaceAndSecurityIndex_Score_2023',
                     title="Top 10 Countries by WPS Index")
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation Analysis
    st.header("Correlation Analysis")
    fig3 = px.imshow(stats['correlation_matrix'],
                     title="Correlation Matrix of Safety Metrics")
    st.plotly_chart(fig3, use_container_width=True)

    # Clustering Analysis
    st.header("Country Clustering Analysis")
    if plot_data and 'scatter_data' in plot_data:
        # Prepare data for clustering
        clustering_data = plot_data['scatter_data'].copy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clustering_data['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create scatter plot with clusters
        fig4 = px.scatter(clustering_data,
                         x='WomenPeaceAndSecurityIndex_Score_2023',
                         y='WDIStreetSafety_2019',
                         color='Cluster',
                         title="Country Clusters Based on Safety Metrics")
        st.plotly_chart(fig4, use_container_width=True)

    # Violence Metrics Comparison
    st.header("Violence Metrics Comparison")
    if plot_data and 'violence_data' in plot_data:
        fig5 = px.box(plot_data['violence_data'],
                      x='Metric',
                      y='Score',
                      title="Distribution of Violence Metrics Across Countries")
        st.plotly_chart(fig5, use_container_width=True)

    # Summary Statistics
    st.header("Summary Statistics")
    st.dataframe(stats['summary_stats'])

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

else:
    st.error("Failed to load data. Please check if the data file exists and is correctly formatted.")
