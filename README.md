# climate_visualization.py
# Required libraries: pandas, numpy, seaborn, matplotlib, plotly, cartopy, dask
# Install with: pip install pandas numpy seaborn matplotlib plotly cartopy dask

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from cartopy import crs as ccrs
import dask.dataframe as dd

# ----------------------------
# 1. Data Loading & Preprocessing
# ----------------------------
def load_and_preprocess(filepath):
    """Load and prepare NASA GISS temperature data"""
    df = pd.read_csv(filepath)
    
    # Clean and interpolate
    df['Anomaly'] = df['Anomaly'].interpolate(limit=3)
    
    # Calculate decadal averages
    df['Decade'] = (df['Year'] // 10) * 10
    decadal_avg = df.groupby(['Latitude', 'Longitude', 'Decade'])['Anomaly'].mean().reset_index()
    
    return df, decadal_avg

# ----------------------------
# 2. Static Heatmap Visualization
# ----------------------------
def create_static_heatmap(decadal_data):
    """Generate static seaborn heatmap"""
    heatmap_data = decadal_data.pivot_table(index='Latitude', 
                                          columns='Longitude', 
                                          values='Anomaly')

    plt.figure(figsize=(18, 8))
    sns.heatmap(heatmap_data, 
                cmap='coolwarm',
                cbar_kws={'label': 'Temperature Anomaly (°C)'})
    plt.title('Global Temperature Anomalies by Geographic Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# ----------------------------
# 3. Interactive Animation
# ----------------------------
def create_interactive_animation(decadal_data):
    """Generate animated Plotly visualization"""
    fig = px.density_mapbox(decadal_data,
                          lat='Latitude',
                          lon='Longitude',
                          z='Anomaly',
                          animation_frame='Decade',
                          radius=15,
                          center=dict(lat=0, lon=0), 
                          zoom=1,
                          mapbox_style='carto-positron',
                          title='Decadal Temperature Anomaly Progression')
    fig.show()

# ----------------------------
# 4. Projected Geospatial Plot
# ----------------------------
def create_projected_map(raw_data):
    """Create cartopy projected visualization"""
    lons = raw_data['Longitude'].unique()
    lats = raw_data['Latitude'].unique()
    temp_grid = raw_data.pivot_table(index='Latitude', 
                                    columns='Longitude', 
                                    values='Anomaly').values

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines()
    cont = ax.contourf(lons, lats, temp_grid,
                      transform=ccrs.PlateCarree(),
                      cmap='viridis')
    plt.colorbar(cont, label='Temperature Anomaly (°C)')
    plt.title('Global Temperature Anomalies (Robinson Projection)')
    plt.show()

# ----------------------------
# 5. Large Data Processing
# ----------------------------
def process_large_data(filepath):
    """Handle big datasets with Dask"""
    ddf = dd.read_csv(filepath)
    ddf['Decade'] = (ddf['Year'] // 10) * 10
    return ddf.groupby(['Latitude', 'Longitude', 'Decade'])['Anomaly'].mean().compute()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # File path adjustment needed
    data_path = 'NASA_GISS_Global_Temp.csv'
    
    # Process data
    raw_df, decade_avg = load_and_preprocess(data_path)
    
    # Generate visualizations
    create_static_heatmap(decade_avg)
    create_interactive_animation(decade_avg)
    create_projected_map(raw_df)
    
    # For large datasets (>10GB)
    # large_data = process_large_data(data_path)
