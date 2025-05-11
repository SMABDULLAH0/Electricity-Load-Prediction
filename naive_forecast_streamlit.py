import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Cache data loading
@st.cache_data
def load_data(file_path):
    """Load and validate dataset."""
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        numeric_cols = ['demand', 'demand_next_day', 'temperature', 'humidity', 'windSpeed', 'pressure', 'precipIntensity', 'hour', 'day_of_week', 'month']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df['Timestamp'].isnull().any() or df[numeric_cols].isnull().any().sum() > 0.1 * len(df):
            st.error("Dataset contains excessive missing or invalid data.")
            return None
        return df
    except FileNotFoundError:
        st.error("Error: 'data_nextday.csv' not found.")
        return None

# Cache preprocessing
@st.cache_data
def preprocess_data(df):
    """Preprocess data with anomaly detection and feature engineering."""
    df = df.copy()
    # Anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    features = ['temperature', 'humidity', 'windSpeed', 'demand']
    df['Anomaly'] = iso_forest.fit_predict(df[features].fillna(df[features].mean()))
    df = df[df['Anomaly'] == 1].drop(columns=['Anomaly'])
    
    # Feature engineering: lagged demand and interaction terms
    df['demand_lag24'] = df['demand'].shift(24)
    df['temp_demand_interaction'] = df['temperature'] * df['demand']
    
    return df.dropna()

# Cache clustering analysis
@st.cache_data
def clustering_analysis(df, cluster_features, k_range):
    """Perform clustering analysis with elbow and silhouette plots."""
    inertias = []
    silhouette_scores = []
    X = StandardScaler().fit_transform(df[cluster_features])
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        labels = kmeans.predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    return inertias, silhouette_scores

# Cache clustering
@st.cache_data
def preprocess_and_cluster(df, city, start_date, end_date, k_clusters):
    """Preprocess and cluster data with descriptive labels."""
    filtered_df = df[(df['city'] == city) & (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)].copy()
    if filtered_df.empty:
        return None, None, None, None, None
    
    cluster_features = ['temperature', 'humidity', 'windSpeed', 'demand']
    temporal_features = ['hour', 'month']
    filtered_df = filtered_df.dropna(subset=cluster_features + temporal_features)
    
    if filtered_df.empty:
        return None, None, None, None, None
    
    # Scale features
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(filtered_df[cluster_features])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_cluster)
    filtered_df['PCA1'] = pca_data[:, 0]
    filtered_df['PCA2'] = pca_data[:, 1]
    
    # Assign descriptive labels
    filtered_df, cluster_labels, centroids = assign_cluster_labels(filtered_df, cluster_features, temporal_features, k_clusters)
    
    return filtered_df, kmeans, scaler, cluster_labels, centroids

def assign_cluster_labels(df, cluster_features, temporal_features, k_clusters):
    """Assign descriptive labels to clusters."""
    centroids = df.groupby('Cluster')[cluster_features + temporal_features].mean()
    labels = {}
    
    temp_high = centroids['temperature'].quantile(0.75)
    temp_low = centroids['temperature'].quantile(0.25)
    humidity_high = centroids['humidity'].quantile(0.75)
    wind_high = centroids['windSpeed'].quantile(0.75)
    demand_high = centroids['demand'].quantile(0.75)
    demand_low = centroids['demand'].quantile(0.25)
    
    for cluster_id in range(k_clusters):
        centroid = centroids.loc[cluster_id]
        temp = centroid['temperature']
        humidity = centroid['humidity']
        wind = centroid['windSpeed']
        demand = centroid['demand']
        hour = centroid['hour']
        month = centroid['month']
        
        label = ""
        if temp >= temp_high and month in [6, 7, 8]:
            label += "Summer "
        elif temp <= temp_low and month in [12, 1, 2]:
            label += "Winter "
        elif month in [3, 4, 5]:
            label += "Spring "
        elif month in [9, 10, 11]:
            label += "Fall "
        
        if 12 <= hour <= 18:
            label += "Afternoon"
        elif 20 <= hour or hour <= 4:
            label += "Night"
        elif 4 < hour <= 8:
            label += "Early Morning"
        else:
            label += "Day"
        
        if humidity >= humidity_high:
            label += " Humid"
        if wind >= wind_high:
            label += " Windy"
        if demand >= demand_high:
            label += " High-Demand"
        elif demand <= demand_low:
            label += " Low-Demand"
        
        labels[cluster_id] = label.strip() or f"Cluster {cluster_id}"
    
    df['Cluster_Label'] = df['Cluster'].map(labels)
    return df, labels, centroids

# Cache forecasting
@st.cache_data
def train_forecast_model(filtered_df, model_type, lookback_window, features):
    """Train forecasting model with hyperparameter tuning."""
    filtered_df = filtered_df.copy()
    filtered_df = filtered_df.dropna(subset=['demand', 'demand_next_day'] + features)
    if filtered_df.empty:
        return None, None, None, None
    
    X = []
    y = []
    for i in range(lookback_window, len(filtered_df) - 24):
        X.append(filtered_df[features].iloc[i - lookback_window:i].values.flatten())
        y.append(filtered_df['demand_next_day'].iloc[i])
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 10:
        return None, None, None, None
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    y_pred = np.zeros_like(y_test)
    metrics = {}
    feature_importance = None
    
    with st.spinner(f"Training {model_type} model..."):
        if model_type == 'Naive':
            y_pred = filtered_df['demand'].shift(24).iloc[train_size:train_size + len(y_test)].values
        elif model_type == 'Linear Regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif model_type == 'Random Forest':
            model = GridSearchCV(RandomForestRegressor(random_state=42), 
                               {'n_estimators': [50, 100], 'max_depth': [10, None]},
                               cv=3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            feature_importance = pd.DataFrame({
                'Feature': [f"{f}_t-{i}" for f in features for i in range(lookback_window-1, -1, -1)],
                'Importance': model.best_estimator_.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif model_type == 'XGBoost':
            model = GridSearchCV(XGBRegressor(random_state=42),
                               {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
                               cv=3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            feature_importance = pd.DataFrame({
                'Feature': [f"{f}_t-{i}" for f in features for i in range(lookback_window-1, -1, -1)],
                'Importance': model.best_estimator_.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif model_type == 'SARIMA':
            model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
            model_fit = model.fit(disp=False)
            y_pred = model_fit.forecast(steps=len(y_test))
        elif model_type == 'LSTM':
            X_train_lstm = X_train.reshape((X_train.shape[0], lookback_window, len(features)))
            X_test_lstm = X_test.reshape((X_test.shape[0], lookback_window, len(features)))
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(lookback_window, len(features))),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_lstm, y_train, epochs=3, batch_size=32, verbose=0)
            y_pred = model.predict(X_test_lstm, verbose=0).flatten()
        elif model_type == 'Stacking Ensemble':
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('xgb', XGBRegressor(n_estimators=50, random_state=42)),
                ('lr', LinearRegression())
            ]
            model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    
    if len(y_test) > 0 and len(y_pred) > 0:
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE': (abs(y_test - y_pred) / y_test).mean() * 100
        }
    
    return y_test, y_pred, metrics, feature_importance

# Generate PDF report
def generate_pdf_report(filtered_df, centroids, metrics_dict, k_clusters, model_type):
    """Generate a PDF report summarizing results."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    c.drawString(50, 750, "Electric Load Forecasting Project Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clustering Summary
    c.drawString(50, 700, "Clustering Summary")
    c.drawString(50, 680, f"Number of Clusters (k): {k_clusters}")
    c.drawString(50, 660, "Cluster Characteristics:")
    y_pos = 640
    for cluster_id, label in filtered_df[['Cluster', 'Cluster_Label']].drop_duplicates().set_index('Cluster')['Cluster_Label'].items():
        centroid = centroids.loc[cluster_id]
        c.drawString(50, y_pos, f"{label}:")
        c.drawString(70, y_pos-15, f"Temp: {centroid['temperature']:.1f}°F, Humidity: {centroid['humidity']:.1f}%, "
                     f"Wind: {centroid['windSpeed']:.1f}mph, Demand: {centroid['demand']:.1f}MWh")
        c.drawString(70, y_pos-30, f"Hour: {centroid['hour']:.1f}, Month: {centroid['month']:.1f}")
        y_pos -= 50
    
    # Forecasting Summary
    c.drawString(50, y_pos, "Forecasting Summary")
    c.drawString(50, y_pos-20, f"Selected Model: {model_type}")
    for model, metrics in metrics_dict.items():
        c.drawString(50, y_pos-40, f"{model}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
        y_pos -= 20
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit app
st.title('Electric Load Forecasting Dashboard')
st.write('Advanced forecasting and clustering for next-day electricity demand.')

# Layout with tabs
tab1, tab2, tab3 = st.tabs(["Input & Results", "Clustering Analysis", "Model Insights"])

with tab1:
    # Input Form
    st.sidebar.header('Input Parameters')
    city = st.sidebar.selectbox('Select City', load_data('data_nextday.csv')['city'].unique())
    start_date = st.sidebar.date_input('Start Date', min_value=pd.to_datetime(load_data('data_nextday.csv')['Timestamp']).min().date())
    end_date = st.sidebar.date_input('End Date', min_value=start_date)
    lookback_window = st.sidebar.slider('Look-back Window (hours)', min_value=1, max_value=48, value=24)
    k_clusters = st.sidebar.slider('Number of Clusters (k)', min_value=2, max_value=10, value=4)
    model_type = st.sidebar.selectbox('Select Model', ['Naive', 'Linear Regression', 'Random Forest', 'XGBoost', 'SARIMA', 'LSTM', 'Stacking Ensemble'])
    
    # Load and preprocess data
    df = load_data('data_nextday.csv')
    if df is None:
        st.stop()
    
    df = preprocess_data(df)
    
    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Clustering
    filtered_df, kmeans, scaler, cluster_labels, centroids = preprocess_and_cluster(df, city, start_date, end_date, k_clusters)
    if filtered_df is None:
        st.error("No data available for the selected city and date range.")
        st.stop()
    
    # Forecasting
    features = ['temperature', 'humidity', 'windSpeed', 'pressure', 'precipIntensity', 'hour', 'day_of_week', 'month', 'demand_lag24', 'temp_demand_interaction']
    metrics_dict = {}
    for model in ['Naive', 'Linear Regression', 'Random Forest', 'XGBoost', 'SARIMA', 'LSTM', 'Stacking Ensemble']:
        y_test, y_pred, metrics, _ = train_forecast_model(filtered_df, model, lookback_window, features)
        if y_test is not None:
            metrics_dict[model] = metrics
    
    if not metrics_dict:
        st.error("Insufficient data for forecasting. Try expanding the date range.")
        st.stop()
    
    y_test, y_pred, metrics, feature_importance = train_forecast_model(filtered_df, model_type, lookback_window, features)
    
    # Results Display
    st.header('Results')
    
    # Cluster Visualization
    st.subheader('Cluster Visualization (PCA)')
    fig = px.scatter(filtered_df, x='PCA1', y='PCA2', color='Cluster_Label', hover_data=['Timestamp', 'demand'],
                     title='Consumption-Weather Clusters', color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Summary
    st.subheader('Cluster Characteristics')
    centroids_df = centroids.reset_index()
    centroids_df['Label'] = centroids_df['Cluster'].map(cluster_labels)
    st.table(centroids_df[['Label', 'temperature', 'humidity', 'windSpeed', 'demand', 'hour', 'month']])
    
    # Forecast Plot
    st.subheader('Forecast vs Actual Demand')
    timestamps = filtered_df['Timestamp'].iloc[-len(y_test):]
    plot_df = pd.DataFrame({'Timestamp': timestamps, 'Actual': y_test, 'Predicted': y_pred})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Timestamp'], y=plot_df['Actual'], mode='lines', name='Actual Demand', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_df['Timestamp'], y=plot_df['Predicted'], mode='lines', name='Predicted Demand', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title=f'{model_type} Forecast vs Actual Demand',
        xaxis_title='Timestamp',
        yaxis_title='Demand (MWh)',
        showlegend=True,
        xaxis_tickangle=45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    st.subheader('Model Performance')
    for model, metrics in metrics_dict.items():
        st.write(f"{model}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

with tab2:
    # Clustering Analysis
    st.header('Clustering Analysis')
    cluster_features = ['temperature', 'humidity', 'windSpeed', 'demand']
    k_range = range(2, 11)
    inertias, silhouette_scores = clustering_analysis(filtered_df, cluster_features, k_range)
    
    # Elbow Plot
    st.subheader('Elbow Method')
    fig, ax = plt.subplots()
    ax.plot(k_range, inertias, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)
    
    # Silhouette Plot
    st.subheader('Silhouette Scores')
    fig, ax = plt.subplots()
    ax.plot(k_range, silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Optimal k')
    st.pyplot(fig)

with tab3:
    # Model Insights
    st.header('Model Insights')
    if feature_importance is not None and model_type in ['Random Forest', 'XGBoost']:
        st.subheader('Feature Importance')
        fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', title=f'{model_type} Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# Help & Documentation
with st.expander('Help & Documentation'):
    st.write("""
    ### Instructions
    - **Select City**: Choose a city from the dropdown.
    - **Date Range**: Pick start and end dates for analysis.
    - **Look-back Window**: Adjust the number of hours used as input for forecasting.
    - **Number of Clusters (k)**: Set the number of clusters for K-Means.
    - **Model Type**: Choose a forecasting model.
    
    ### Preprocessing
    - **Anomaly Detection**: Isolation Forest removes 10% of outliers.
    - **Feature Engineering**: Includes lagged demand and temperature-demand interaction.
    
    ### Clustering
    - **Algorithm**: K-Means with PCA visualization.
    - **Evaluation**: Elbow method and silhouette scores guide k selection.
    - **Labels**: Descriptive labels based on temperature, humidity, wind speed, demand, hour, and month.
    
    ### Forecasting
    - **Models**:
      - **Naive**: Previous day's demand.
      - **Linear Regression**: Linear model.
      - **Random Forest**: Ensemble with grid search.
      - **XGBoost**: Gradient boosting with grid search.
      - **SARIMA**: Seasonal ARIMA with 24-hour cycle.
      - **LSTM**: Neural network for sequential data.
      - **Stacking Ensemble**: Combines RF, XGBoost, and LR.
    - **Metrics**: MAE, RMSE, MAPE.
    
    ### Technical Details
    - **Data Source**: Kaggle dataset with hourly demand and weather data.
    - **Dependencies**: pandas, numpy, scikit-learn, xgboost, statsmodels, tensorflow, plotly, streamlit, reportlab.
    """)

# Download Results
if st.button('Download Results'):
    filtered_df.to_csv('clustered_data.csv', index=False)
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv('forecast_metrics.csv', index=False)
    centroids_df.to_csv('cluster_centroids.csv', index=False)
    pdf_buffer = generate_pdf_report(filtered_df, centroids, metrics_dict, k_clusters, model_type)
    st.download_button("Download PDF Report", pdf_buffer, "forecast_report.pdf", "application/pdf")
    st.success('Files saved: clustered_data.csv, forecast_metrics.csv, cluster_centroids.csv, forecast_report.pdf')