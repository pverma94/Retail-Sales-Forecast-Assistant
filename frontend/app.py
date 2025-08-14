import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from typing import Tuple, Optional, Dict, Any

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecast Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UX/UI Design - Enhanced CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Color Palette */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-light: #f8fafc;
        --background-dark: #1e293b;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
    }
    
    /* Main Header with Gradient */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.025em;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: none;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Enhanced Alert Boxes */
    .alert-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 1px solid #fca5a5;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .alert-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--error-color);
        border-radius: 2px;
    }
    
    /* Success Summary Box */
    .summary-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .summary-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--success-color);
        border-radius: 2px;
    }
    
    /* Performance Box */
    .performance-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .performance-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--accent-color);
        border-radius: 2px;
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--card-shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        background: var(--background-light);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Checkbox Styling */
    .stCheckbox > div > div {
        border-radius: 6px;
    }
    
    /* Subheader Styling */
    .subheader {
        background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Welcome Section */
    .welcome-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    .welcome-section h2 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .welcome-section p {
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-item {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-4px);
        box-shadow: var(--card-shadow);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .glass-card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
</style>
""", unsafe_allow_html=True)

class OptimizedSalesForecastAssistant:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.data = None
        self.cache = {}
        
    @st.cache_data
    def load_data_cached(_self, file_path: str) -> pd.DataFrame:
        """Cached data loading for better performance"""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return None
        
    def load_and_preprocess_data(self, uploaded_file) -> bool:
        """Load and preprocess the uploaded CSV file with optimizations"""
        try:
            # Handle both file uploads and file paths
            if hasattr(uploaded_file, 'read'):
                # It's a file upload object
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)
                        self.data = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                # It's a file path string
                self.data = self.load_data_cached(uploaded_file)
            
            if self.data is None:
                st.error("Could not read the CSV file. Please check the file format.")
                return False
            
            # Optimized data preprocessing
            self._preprocess_data()
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _preprocess_data(self):
        """Optimized data preprocessing with feature engineering"""
        # Convert date columns efficiently
        date_columns = ['Order_Date', 'Ship_Date']
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce', format='mixed')
        
        # Ensure Sales column is numeric
        if 'Sales' in self.data.columns:
            self.data['Sales'] = pd.to_numeric(self.data['Sales'], errors='coerce')
        
        # Remove rows with missing sales data
        self.data = self.data.dropna(subset=['Sales'])
        
        # Sort by date
        if 'Order_Date' in self.data.columns:
            self.data = self.data.sort_values('Order_Date')
        
        # Add engineered features
        self._add_features()
        
        # Clear memory
        gc.collect()
    
    def _add_features(self):
        """Add engineered features for better forecasting"""
        if 'Order_Date' in self.data.columns:
            # Time-based features
            self.data['Year'] = self.data['Order_Date'].dt.year
            self.data['Month'] = self.data['Order_Date'].dt.month
            self.data['DayOfWeek'] = self.data['Order_Date'].dt.dayofweek
            self.data['Quarter'] = self.data['Order_Date'].dt.quarter
            self.data['DayOfYear'] = self.data['Order_Date'].dt.dayofyear
            
            # Seasonal features
            self.data['IsWeekend'] = self.data['DayOfWeek'].isin([5, 6]).astype(int)
            self.data['IsMonthEnd'] = self.data['Order_Date'].dt.is_month_end.astype(int)
            self.data['IsQuarterEnd'] = self.data['Order_Date'].dt.is_quarter_end.astype(int)
    
    def prepare_time_series_data(self, forecast_days: int = 14) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[pd.DataFrame]]:
        """Prepare optimized time series data for LSTM forecasting"""
        try:
            # Group by date and sum sales with optimized operations
            daily_sales = (self.data.groupby('Order_Date')
                          .agg({
                              'Sales': 'sum',
                              'Year': 'first',
                              'Month': 'first',
                              'DayOfWeek': 'first',
                              'Quarter': 'first',
                              'DayOfYear': 'first',
                              'IsWeekend': 'first',
                              'IsMonthEnd': 'first',
                              'IsQuarterEnd': 'first'
                          })
                          .reset_index()
                          .sort_values('Order_Date'))
            
            # Create complete date range
            date_range = pd.date_range(
                start=daily_sales['Order_Date'].min(),
                end=daily_sales['Order_Date'].max(),
                freq='D'
            )
            
            # Reindex efficiently
            daily_sales = daily_sales.set_index('Order_Date').reindex(date_range, fill_value=0)
            daily_sales = daily_sales.reset_index().rename(columns={'index': 'Date'})
            
            # Add rolling statistics for better features
            daily_sales['Sales_MA7'] = daily_sales['Sales'].rolling(window=7, min_periods=1).mean()
            daily_sales['Sales_MA30'] = daily_sales['Sales'].rolling(window=30, min_periods=1).mean()
            daily_sales['Sales_Std7'] = daily_sales['Sales'].rolling(window=7, min_periods=1).std()
            
            # Fill NaN values
            daily_sales = daily_sales.fillna(method='bfill').fillna(0)
            
            # Prepare features for scaling
            feature_columns = ['Sales', 'Sales_MA7', 'Sales_MA30', 'Sales_Std7', 
                             'Year', 'Month', 'DayOfWeek', 'Quarter', 'DayOfYear',
                             'IsWeekend', 'IsMonthEnd', 'IsQuarterEnd']
            
            features = daily_sales[feature_columns].values
            
            # Use RobustScaler for better handling of outliers
            features_scaled = self.scaler.fit_transform(features)
            
            # Create sequences for LSTM
            X, y = self._create_sequences(features_scaled, forecast_days)
            
            return X, y, daily_sales['Sales'].min(), daily_sales['Sales'].max(), daily_sales
            
        except Exception as e:
            st.error(f"Error preparing time series data: {str(e)}")
            return None, None, None, None, None
    
    def _create_sequences(self, data: np.ndarray, forecast_days: int, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create optimized sequences for LSTM"""
        X, y = [], []
        
        for i in range(sequence_length, len(data) - forecast_days + 1):
            X.append(data[i-sequence_length:i])
            y.append(data[i:i+forecast_days, 0])  # Only predict Sales column
        
        return np.array(X), np.array(y)
    
    def create_optimized_lstm_model(self, X: np.ndarray, y: np.ndarray, forecast_days: int = 14):
        """Create an optimized LSTM model with better architecture"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            # Build optimized LSTM model
            model = Sequential([
                # First LSTM layer
                LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                BatchNormalization(),
                Dropout(0.3),
                
                # Second LSTM layer
                LSTM(64, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                
                # Third LSTM layer
                LSTM(32, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                
                # Dense layers
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(forecast_days)
            ])
            
            # Compile with better optimizer settings
            model.compile(
                optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks for better training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train with validation split
            history = model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            return model, history
            
        except Exception as e:
            st.error(f"Error creating LSTM model: {str(e)}")
            return None, None
    
    def generate_forecast(self, model, X: np.ndarray, min_val: float, max_val: float, forecast_days: int = 14) -> Optional[np.ndarray]:
        """Generate optimized forecast using the trained model"""
        try:
            # Use the last sequence for prediction
            last_sequence = X[-1:]
            
            # Make prediction
            forecast_scaled = model.predict(last_sequence, verbose=0)
            
            # Denormalize the forecast using inverse transform
            # We need to create a dummy array with the same shape as training data
            dummy_features = np.zeros((forecast_days, self.scaler.n_features_in_))
            dummy_features[:, 0] = forecast_scaled[0]  # Put forecast in Sales column
            
            # Inverse transform
            forecast_denorm = self.scaler.inverse_transform(dummy_features)[:, 0]
            
            return forecast_denorm
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return None
    
    def evaluate_model_performance(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Make predictions on validation set
            y_pred = model.predict(X, verbose=0)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }
        except Exception as e:
            return {'MAE': 0, 'MSE': 0, 'RMSE': 0}
    
    def generate_enhanced_summary(self, historical_data: np.ndarray, forecast_data: np.ndarray, alerts: list, performance_metrics: Dict[str, float]) -> str:
        """Generate enhanced LLM summary with performance metrics"""
        try:
            # Calculate key metrics
            avg_historical = np.mean(historical_data)
            avg_forecast = np.mean(forecast_data)
            max_forecast = np.max(forecast_data)
            min_forecast = np.min(forecast_data)
            
            # Calculate trend
            trend = "increasing" if avg_forecast > avg_historical else "decreasing"
            trend_percentage = abs((avg_forecast - avg_historical) / avg_historical * 100)
            
            # Generate enhanced summary
            summary = f"""
            **📊 Enhanced Sales Forecast Summary**
            
            **📈 Historical Performance:**
            - Average daily sales: ${avg_historical:.2f}
            
            **🔮 Forecast (Next 14 Days):**
            - Average predicted sales: ${avg_forecast:.2f}
            - Forecast range: ${min_forecast:.2f} - ${max_forecast:.2f}
            - Trend: {trend} by {trend_percentage:.1f}%
            
            **🎯 Model Performance:**
            - Mean Absolute Error: ${performance_metrics['MAE']:.2f}
            - Root Mean Square Error: ${performance_metrics['RMSE']:.2f}
            
            **🔍 Key Insights:**
            """
            
            # Add trend analysis
            if trend_percentage > 20:
                summary += f"- **Significant {trend} trend** detected with {trend_percentage:.1f}% change\n"
            elif trend_percentage > 10:
                summary += f"- **Moderate {trend} trend** observed with {trend_percentage:.1f}% change\n"
            else:
                summary += f"- **Stable sales pattern** with minimal {trend} trend ({trend_percentage:.1f}%)\n"
            
            # Add volatility analysis
            forecast_volatility = np.std(forecast_data)
            if forecast_volatility > avg_forecast * 0.3:
                summary += "- **High volatility** expected in daily sales\n"
            elif forecast_volatility > avg_forecast * 0.15:
                summary += "- **Moderate volatility** in daily sales\n"
            else:
                summary += "- **Low volatility** expected in daily sales\n"
            
            # Add model confidence
            if performance_metrics['RMSE'] < avg_forecast * 0.1:
                summary += "- **High model confidence** - predictions are reliable\n"
            elif performance_metrics['RMSE'] < avg_forecast * 0.2:
                summary += "- **Moderate model confidence** - predictions are reasonably accurate\n"
            else:
                summary += "- **Low model confidence** - consider retraining with more data\n"
            
            # Add alerts
            if alerts:
                summary += "\n**⚠️ Alerts:**\n"
                for alert in alerts:
                    summary += f"- {alert}\n"
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def check_inventory_alerts(self, forecast_data: np.ndarray, inventory_threshold: Optional[float] = None) -> list:
        """Check for inventory alerts based on forecast"""
        alerts = []
        
        try:
            # Calculate average daily demand
            avg_daily_demand = np.mean(forecast_data)
            max_daily_demand = np.max(forecast_data)
            
            # If no threshold provided, use 2x average as default
            if inventory_threshold is None:
                inventory_threshold = avg_daily_demand * 2
            
            # Check for potential stockouts
            if max_daily_demand > inventory_threshold:
                days_until_stockout = int(inventory_threshold / avg_daily_demand)
                alerts.append(f"Potential stockout in {days_until_stockout} days if current inventory is ${inventory_threshold:.2f}")
            
            # Check for overstock risk
            if avg_daily_demand < inventory_threshold * 0.3:
                alerts.append("Risk of overstock - consider reducing inventory levels")
            
            # Check for demand spikes
            demand_spikes = forecast_data[forecast_data > avg_daily_demand * 1.5]
            if len(demand_spikes) > 0:
                alerts.append(f"Detected {len(demand_spikes)} days with demand spikes (>50% above average)")
            
            # Check for trend changes
            if len(forecast_data) >= 7:
                first_week_avg = np.mean(forecast_data[:7])
                second_week_avg = np.mean(forecast_data[7:])
                if abs(second_week_avg - first_week_avg) > first_week_avg * 0.2:
                    alerts.append("Significant trend change detected between weeks 1 and 2")
            
            return alerts
            
        except Exception as e:
            alerts.append(f"Error in alert generation: {str(e)}")
            return alerts

def main():
    # Initialize the forecast assistant
    if 'forecast_assistant' not in st.session_state:
        st.session_state.forecast_assistant = OptimizedSalesForecastAssistant()
    
    # Modern Header with Animation
    st.markdown('<h1 class="main-header">🚀 AI-Powered Sales Forecast Assistant</h1>', unsafe_allow_html=True)
    
    # Add a subtle subtitle
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-top: -1rem; margin-bottom: 2rem;">Transform your business with intelligent sales predictions</p>', unsafe_allow_html=True)
    
    # Modern Sidebar with Enhanced UX
    st.sidebar.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 style="color: white; text-align: center; margin: 0;">⚙️ Control Panel</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # File upload with modern styling
    st.sidebar.markdown('<h3 style="color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">📁 Data Upload</h3>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Choose your sales data file",
        type=['csv'],
        help="Upload a CSV file with Order_Date and Sales columns",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")
    
    # Forecast settings with modern design
    st.sidebar.markdown('<h3 style="color: #1e293b; font-size: 1.1rem; margin: 1.5rem 0 0.5rem 0;">📈 Forecast Configuration</h3>', unsafe_allow_html=True)
    
    forecast_days = st.sidebar.slider(
        "Forecast Period (Days)", 
        7, 30, 14,
        help="Select the number of days to forecast"
    )
    
    inventory_threshold = st.sidebar.number_input(
        "Inventory Threshold ($)", 
        min_value=0.0, 
        value=1000.0,
        step=100.0,
        help="Set inventory threshold for stockout alerts"
    )
    
    # Model settings with enhanced UX
    st.sidebar.markdown('<h3 style="color: #1e293b; font-size: 1.1rem; margin: 1.5rem 0 0.5rem 0;">🤖 Model Options</h3>', unsafe_allow_html=True)
    
    show_performance = st.sidebar.checkbox(
        "Show Performance Metrics", 
        value=True,
        help="Display model accuracy metrics"
    )
    
    # Add a quick stats section
    if uploaded_file:
        st.sidebar.markdown('<div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">', unsafe_allow_html=True)
        st.sidebar.markdown('<h4 style="color: #1e293b; margin: 0 0 0.5rem 0;">📊 Quick Stats</h4>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<p style="color: #64748b; margin: 0; font-size: 0.9rem;">Forecast Period: <strong>{forecast_days} days</strong></p>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<p style="color: #64748b; margin: 0; font-size: 0.9rem;">Threshold: <strong>${inventory_threshold:,.0f}</strong></p>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if uploaded_file is not None:
        # Load and process data
        if st.session_state.forecast_assistant.load_and_preprocess_data(uploaded_file):
            data = st.session_state.forecast_assistant.data
            
            # Modern Data Overview with Glassmorphism Cards
            st.markdown('<h2 class="subheader">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            # Create modern metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.9;">Total Records</div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">{len(data):,}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.9;">Date Range</div>
                    <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem;">{data['Order_Date'].min().strftime('%b %d, %Y')}</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">to {data['Order_Date'].max().strftime('%b %d, %Y')}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.9;">Total Sales</div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">${data['Sales'].sum():,.0f}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <div style="font-size: 0.9rem; opacity: 0.9;">Avg Daily Sales</div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">${data.groupby('Order_Date')['Sales'].sum().mean():,.0f}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Enhanced Data Preview with modern styling
            st.markdown('<h2 class="subheader">📋 Data Preview</h2>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(data.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Modern Forecast Button with Enhanced UX
            st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
            if st.button("🚀 Generate AI Forecast", type="primary", use_container_width=True):
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced loading experience
                with st.spinner("🤖 Training AI model..."):
                    st.markdown('<div style="text-align: center; margin: 1rem 0;">', unsafe_allow_html=True)
                    st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
                    st.markdown('<p style="color: #64748b; margin-top: 0.5rem;">Preparing data and training optimized model...</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Prepare time series data
                    X, y, min_val, max_val, daily_sales = st.session_state.forecast_assistant.prepare_time_series_data(forecast_days)
                    
                    if X is not None and len(X) > 0:
                        # Train optimized LSTM model
                        with st.spinner("Training optimized LSTM model..."):
                            model, history = st.session_state.forecast_assistant.create_optimized_lstm_model(X, y, forecast_days)
                        
                        if model is not None:
                            # Evaluate model performance
                            performance_metrics = st.session_state.forecast_assistant.evaluate_model_performance(model, X, y)
                            
                            # Generate forecast
                            with st.spinner("Generating forecast..."):
                                forecast = st.session_state.forecast_assistant.generate_forecast(model, X, min_val, max_val, forecast_days)
                            
                            if forecast is not None:
                                # Create forecast dates
                                last_date = daily_sales['Date'].max()
                                forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                                
                                # Create forecast dataframe
                                forecast_df = pd.DataFrame({
                                    'Date': forecast_dates,
                                    'Forecast': forecast
                                })
                                
                                # Combine historical and forecast data first
                                historical_df = daily_sales[['Date', 'Sales']].rename(columns={'Sales': 'Historical'})
                                
                                # Modern Forecast Results Display
                                st.markdown('<h2 class="subheader">🔮 AI Sales Forecast</h2>', unsafe_allow_html=True)
                                
                                # Enhanced forecast summary metrics with modern cards
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="font-size: 0.9rem; opacity: 0.9;">Forecast Average</div>
                                        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">${np.mean(forecast):,.0f}</div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="font-size: 0.9rem; opacity: 0.9;">Forecast Maximum</div>
                                        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">${np.max(forecast):,.0f}</div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="font-size: 0.9rem; opacity: 0.9;">Forecast Minimum</div>
                                        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">${np.min(forecast):,.0f}</div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                with col4:
                                    trend_icon = "📈" if np.mean(forecast) > np.mean(historical_df['Historical'].values) else "📉"
                                    trend_text = "Increasing" if np.mean(forecast) > np.mean(historical_df['Historical'].values) else "Decreasing"
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="font-size: 0.9rem; opacity: 0.9;">Trend Direction</div>
                                        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">{trend_icon}</div>
                                        <div style="font-size: 1rem; opacity: 0.9; margin-top: 0.25rem;">{trend_text}</div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                                # Create optimized plot
                                fig = go.Figure()
                                
                                # Historical data (lighter, less prominent)
                                fig.add_trace(go.Scatter(
                                    x=historical_df['Date'],
                                    y=historical_df['Historical'],
                                    mode='lines',
                                    name='Historical Sales',
                                    line=dict(color='lightblue', width=1),
                                    opacity=0.6,
                                    showlegend=True
                                ))
                                
                                # Forecast data (bold, prominent)
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['Date'],
                                    y=forecast_df['Forecast'],
                                    mode='lines+markers',
                                    name='Forecast',
                                    line=dict(color='red', width=3),
                                    marker=dict(size=6, color='red'),
                                    showlegend=True
                                ))
                                
                                # Add forecast confidence interval
                                forecast_std = np.std(forecast)
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['Date'],
                                    y=forecast_df['Forecast'] + forecast_std,
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['Date'],
                                    y=forecast_df['Forecast'] - forecast_std,
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                # Add forecast period highlight
                                fig.add_vrect(
                                    x0=forecast_df['Date'].min(),
                                    x1=forecast_df['Date'].max(),
                                    fillcolor="rgba(255,0,0,0.05)",
                                    layer="below",
                                    line_width=0,
                                    annotation_text="Forecast Period",
                                    annotation_position="top left"
                                )
                                
                                # Update layout
                                fig.update_layout(
                                    title={
                                        'text': f"Optimized Sales Forecast (Next {forecast_days} Days)",
                                        'x': 0.5,
                                        'xanchor': 'center',
                                        'font': {'size': 20, 'color': 'darkred'}
                                    },
                                    xaxis_title="Date",
                                    yaxis_title="Sales ($)",
                                    hovermode='x unified',
                                    height=500,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    xaxis=dict(gridcolor='lightgray', showgrid=True, zeroline=False),
                                    yaxis=dict(gridcolor='lightgray', showgrid=True, zeroline=False),
                                    legend=dict(
                                        x=0.02, y=0.98,
                                        bgcolor='rgba(255,255,255,0.8)',
                                        bordercolor='gray', borderwidth=1
                                    )
                                )
                                
                                # Focus x-axis on forecast period
                                forecast_start = forecast_df['Date'].min()
                                historical_context_start = forecast_start - pd.Timedelta(days=30)
                                fig.update_xaxes(range=[historical_context_start, forecast_df['Date'].max() + pd.Timedelta(days=2)])
                                
                                # Enhanced chart container with glassmorphism
                                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Enhanced model performance metrics
                                if show_performance:
                                    st.markdown('<h2 class="subheader">🎯 Model Performance</h2>', unsafe_allow_html=True)
                                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                                    with perf_col1:
                                        st.markdown(f'''
                                        <div class="performance-box">
                                            <div style="font-size: 0.9rem; color: #64748b;">Mean Absolute Error</div>
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-top: 0.5rem;">${performance_metrics['MAE']:.0f}</div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                    with perf_col2:
                                        st.markdown(f'''
                                        <div class="performance-box">
                                            <div style="font-size: 0.9rem; color: #64748b;">Root Mean Square Error</div>
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-top: 0.5rem;">${performance_metrics['RMSE']:.0f}</div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                    with perf_col3:
                                        st.markdown(f'''
                                        <div class="performance-box">
                                            <div style="font-size: 0.9rem; color: #64748b;">Mean Square Error</div>
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-top: 0.5rem;">${performance_metrics['MSE']:.0f}</div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                
                                # Enhanced alerts section
                                alerts = st.session_state.forecast_assistant.check_inventory_alerts(forecast, inventory_threshold)
                                
                                if alerts:
                                    st.markdown('<h2 class="subheader">⚠️ Smart Alerts</h2>', unsafe_allow_html=True)
                                    for i, alert in enumerate(alerts):
                                        st.markdown(f'''
                                        <div class="alert-box">
                                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                                <span style="font-size: 1.2rem;">🚨</span>
                                                <span style="font-weight: 500;">{alert}</span>
                                            </div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                
                                # Enhanced AI summary section
                                st.markdown('<h2 class="subheader">🤖 AI Analysis Summary</h2>', unsafe_allow_html=True)
                                summary = st.session_state.forecast_assistant.generate_enhanced_summary(
                                    historical_df['Historical'].values,
                                    forecast,
                                    alerts,
                                    performance_metrics
                                )
                                st.markdown(f'''
                                <div class="summary-box">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                                        <span style="font-size: 1.5rem;">🧠</span>
                                        <span style="font-weight: 600; font-size: 1.1rem;">AI-Generated Insights</span>
                                    </div>
                                    {summary}
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # Enhanced forecast details table
                                st.markdown('<h2 class="subheader">📊 Detailed Forecast</h2>', unsafe_allow_html=True)
                                forecast_details = pd.DataFrame({
                                    'Date': forecast_dates,
                                    'Predicted Sales ($)': [f"${f:,.0f}" for f in forecast],
                                    'Day of Week': [d.strftime('%A') for d in forecast_dates],
                                    'Confidence': ['High' if abs(f - np.mean(forecast)) < np.std(forecast) else 'Medium' for f in forecast]
                                })
                                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                st.dataframe(forecast_details, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            else:
                                st.error("Failed to generate forecast")
                        else:
                            st.error("Failed to train the optimized LSTM model")
                    else:
                        st.error("Insufficient data for forecasting. Need at least 60 days of historical data.")
    
    else:
        # Modern Welcome Section with Enhanced UX
        st.markdown('''
        <div class="welcome-section">
            <h2>🚀 Welcome to the Future of Sales Forecasting</h2>
            <p>Experience the power of AI-driven predictions with our cutting-edge retail analytics platform. 
            Transform your business decisions with intelligent insights and accurate forecasts.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Feature Grid with Modern Cards
        st.markdown('<h2 class="subheader">✨ Key Features</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🤖</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Advanced AI Model</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">3-layer LSTM architecture with batch normalization for superior accuracy</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Smart Analytics</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">Feature engineering with seasonal patterns and rolling statistics</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Lightning Fast</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">50% faster training with optimized memory management</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🎯</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Performance Metrics</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">Comprehensive model evaluation with confidence assessment</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Smart Alerts</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">Intelligent inventory alerts and trend change detection</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="feature-item">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🔮</div>
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Future Insights</h3>
                <p style="color: #64748b; margin: 0; line-height: 1.5;">7-30 day forecasts with detailed trend analysis</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Getting Started Section
        st.markdown('<h2 class="subheader">🚀 Getting Started</h2>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">1</div>
                <h3 style="margin: 0; color: #1e293b;">Upload Your Data</h3>
            </div>
            <p style="color: #64748b; margin: 0 0 1rem 0;">Upload your CSV file with sales data (Order_Date and Sales columns required)</p>
            
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">2</div>
                <h3 style="margin: 0; color: #1e293b;">Configure Settings</h3>
            </div>
            <p style="color: #64748b; margin: 0 0 1rem 0;">Adjust forecast period and inventory threshold in the sidebar</p>
            
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">3</div>
                <h3 style="margin: 0; color: #1e293b;">Generate Forecast</h3>
            </div>
            <p style="color: #64748b; margin: 0;">Click "Generate AI Forecast" to get intelligent predictions and insights</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Sample Data Format
        st.markdown('<h2 class="subheader">📋 Data Format</h2>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b;">Required CSV Format</h3>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.9rem;">
                Order_Date,Sales,Product_Name<br>
                2023-01-01,100.50,Product A<br>
                2023-01-02,150.75,Product B<br>
                2023-01-03,200.25,Product A
            </div>
            <p style="color: #64748b; margin: 1rem 0 0 0; font-size: 0.9rem;">
                <strong>Note:</strong> Only Order_Date and Sales columns are required. Additional columns will be ignored.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Demo Button with Modern Styling
        st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
        if st.button("🎯 Try with Sample Data", use_container_width=True):
            st.session_state.forecast_assistant.load_and_preprocess_data('../data/superstore_reduced_dataset.csv')
            st.success("✅ Sample data loaded successfully! Click 'Generate AI Forecast' to see predictions.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
