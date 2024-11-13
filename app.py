import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import seaborn as sns

# Configure Streamlit
st.set_page_config(page_title="Women's Safety Research Dashboard", layout="wide")

@st.cache_data
def load_and_prepare_data():
    """Load and prepare data with comprehensive preprocessing"""
    try:
        df = pd.read_csv('most-dangerous-countries-for-women-2024.csv')
        # Clean column names
        df.columns = [col.replace('MostDangerousCountriesForWomen_', '') for col in df.columns]
        
        # Create composite scores
        df['Violence_Score'] = df[['WDIStreetSafety_2019', 'WDIIntentionalHomicide_2019', 
                                 'WDINonPartnerViolence_2019', 'WDIIntimatePartnerViolence_2019']].mean(axis=1)
        
        df['Legal_Social_Score'] = df[['WDILegalDiscrimination_2019', 'WDIGlobalGenderGap_2019', 
                                     'WDIGenderInequality_2019']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def perform_statistical_analysis(df, variable1, variable2):
    """Perform comprehensive statistical analysis between two variables"""
    if df is None or variable1 not in df.columns or variable2 not in df.columns:
        return None
    
    # Remove rows with NaN values for these variables
    clean_data = df[[variable1, variable2]].dropna()
    
    # Calculate correlation coefficient and p-value
    correlation_coef, p_value = stats.pearsonr(clean_data[variable1], clean_data[variable2])
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[variable1], 
                                                                  clean_data[variable2])
    
    return {
        'correlation': correlation_coef,
        'p_value': p_value,
        'r_squared': r_value**2,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }

# Load data
df = load_and_prepare_data()

if df is not None:
    st.title("Women's Safety Research Analysis Dashboard")
    st.write("""
    This research dashboard analyzes relationships between various factors affecting women's safety 
    across different countries. It focuses on identifying correlations, patterns, and potential 
    causative relationships between environmental variables and safety outcomes.
    """)
    
    # Variable Selection for Analysis
    st.sidebar.header("Analysis Parameters")
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create meaningful variable groups
    outcome_vars = ['WomenPeaceAndSecurityIndex_Score_2023', 'Violence_Score', 'Legal_Social_Score']
    predictor_vars = [col for col in numeric_cols if col not in outcome_vars]
    
    # Variable selection
    selected_outcome = st.sidebar.selectbox(
        "Select Outcome Variable",
        outcome_vars,
        help="Choose the safety metric you want to analyze"
    )
    
    selected_predictors = st.sidebar.multiselect(
        "Select Predictor Variables",
        predictor_vars,
        default=predictor_vars[:3],
        help="Choose variables to analyze against the outcome"
    )
    
    # Main Analysis Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Relationship Analysis")
        
        # Create scatter matrix for selected variables
        fig_matrix = px.scatter_matrix(
            df,
            dimensions=[selected_outcome] + selected_predictors,
            title="Relationships Between Selected Variables",
            opacity=0.6
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with col2:
        st.header("Statistical Summary")
        
        for predictor in selected_predictors:
            stats_results = perform_statistical_analysis(df, selected_outcome, predictor)
            
            if stats_results:
                st.subheader(f"{predictor} vs {selected_outcome}")
                st.write(f"Correlation: {stats_results['correlation']:.3f}")
                st.write(f"P-value: {stats_results['p_value']:.3f}")
                st.write(f"R-squared: {stats_results['r_squared']:.3f}")
    
    # Geographic Analysis
    st.header("Geographic Distribution")
    fig_geo = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color=selected_outcome,
        hover_name="country",
        title=f"Global Distribution of {selected_outcome}",
        color_continuous_scale="RdYlBu_r"
    )
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Detailed Country Comparison
    st.header("Country Comparison Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Select countries to compare
        selected_countries = st.multiselect(
            "Select Countries to Compare",
            df['country'].unique(),
            default=df.nlargest(5, selected_outcome)['country'].tolist()
        )
        
        if selected_countries:
            comparison_data = df[df['country'].isin(selected_countries)]
            
            # Create radar chart for selected metrics
            fig_radar = go.Figure()
            
            for country in selected_countries:
                country_data = comparison_data[comparison_data['country'] == country]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=country_data[selected_predictors].values.flatten(),
                    theta=selected_predictors,
                    name=country
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Comparative Analysis of Selected Countries"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col4:
        # Statistical comparison of selected countries
        if selected_countries:
            comparison_df = df[df['country'].isin(selected_countries)][
                ['country', selected_outcome] + selected_predictors
            ]
            st.write("Detailed Metrics Comparison")
            st.dataframe(comparison_df)
    
    # Cluster Analysis
    st.header("Country Clustering Analysis")
    
    # Prepare data for clustering
    cluster_vars = [selected_outcome] + selected_predictors
    cluster_data = df[cluster_vars].dropna()
    
    if not cluster_data.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster information to the dataframe
        cluster_df = df.loc[cluster_data.index].copy()
        cluster_df['Cluster'] = clusters
        
        # Create 3D scatter plot of clusters
        fig_3d = px.scatter_3d(
            cluster_df,
            x=selected_predictors[0] if selected_predictors else cluster_vars[0],
            y=selected_predictors[1] if len(selected_predictors) > 1 else cluster_vars[1],
            z=selected_outcome,
            color='Cluster',
            hover_data=['country'],
            title="3D Cluster Analysis"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Key Findings
    st.header("Key Research Findings")
    
    # Calculate and display significant correlations
    correlation_matrix = df[cluster_vars].corr()
    significant_correlations = []
    
    for i in range(len(cluster_vars)):
        for j in range(i+1, len(cluster_vars)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:  # Threshold for significant correlation
                significant_correlations.append({
                    'var1': cluster_vars[i],
                    'var2': cluster_vars[j],
                    'correlation': corr
                })
    
    if significant_correlations:
        st.subheader("Significant Correlations Found:")
        for corr in significant_correlations:
            st.write(f"• {corr['var1']} and {corr['var2']}: {corr['correlation']:.3f}")
    
    # Download section for research data
    st.header("Download Research Data")
    st.download_button(
        label="Download Full Analysis Results (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='women_safety_research_analysis.csv',
        mime='text/csv',
    )
    
    # Methodology and Notes
    st.markdown("""
    ### Methodology Notes:
    - Correlation analysis uses Pearson's correlation coefficient
    - Clustering is performed using K-means algorithm on standardized data
    - All statistical tests use a significance level of α = 0.05
    - Missing values are handled through pairwise deletion
    """)

else:
    st.error("Failed to load data. Please check if the data file exists and is correctly formatted.")
