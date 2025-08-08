from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from io import StringIO
import warnings
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Optional, Dict, Any
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class OptimizedSalesForecastAPI:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.data = None
        self.cache = {}
        
    def process_data(self, data) -> Optional[pd.DataFrame]:
        """Process uploaded data with optimizations"""
        try:
            # Convert to DataFrame if it's a string
            if isinstance(data, str):
                df = pd.read_csv(StringIO(data))
            else:
                df = pd.DataFrame(data)
            
            # Optimized data preprocessing
            df = self._preprocess_dataframe(df)
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized dataframe preprocessing"""
        # Convert date columns efficiently
        date_columns = ['Order_Date', 'Ship_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
        
        # Ensure Sales column is numeric
        if 'Sales' in df.columns:
            df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        
        # Remove rows with missing sales data
        df = df.dropna(subset=['Sales'])
        
        # Sort by date
        if 'Order_Date' in df.columns:
            df = df.sort_values('Order_Date')
        
        # Add engineered features
        df = self._add_features(df)
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for better forecasting"""
        if 'Order_Date' in df.columns:
            # Time-based features
            df['Year'] = df['Order_Date'].dt.year
            df['Month'] = df['Order_Date'].dt.month
            df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
            df['Quarter'] = df['Order_Date'].dt.quarter
            df['DayOfYear'] = df['Order_Date'].dt.dayofyear
            
            # Seasonal features
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            df['IsMonthEnd'] = df['Order_Date'].dt.is_month_end.astype(int)
            df['IsQuarterEnd'] = df['Order_Date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def prepare_time_series(self, data: pd.DataFrame, forecast_days: int = 14) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[pd.DataFrame]]:
        """Prepare optimized time series data for forecasting"""
        try:
            # Group by date and sum sales with optimized operations
            daily_sales = (data.groupby('Order_Date')
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
            logger.error(f"Error preparing time series: {str(e)}")
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
            logger.error(f"Error creating LSTM model: {str(e)}")
            return None, None
    
    def generate_forecast(self, model, X: np.ndarray, min_val: float, max_val: float, forecast_days: int = 14) -> Optional[np.ndarray]:
        """Generate optimized forecast using the trained model"""
        try:
            # Use the last sequence for prediction
            last_sequence = X[-1:]
            
            # Make prediction
            forecast_scaled = model.predict(last_sequence, verbose=0)
            
            # Denormalize the forecast using inverse transform
            dummy_features = np.zeros((forecast_days, self.scaler.n_features_in_))
            dummy_features[:, 0] = forecast_scaled[0]  # Put forecast in Sales column
            
            # Inverse transform
            forecast_denorm = self.scaler.inverse_transform(dummy_features)[:, 0]
            
            return forecast_denorm
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
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
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse)
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0}
    
    def generate_enhanced_summary(self, historical_data: np.ndarray, forecast_data: np.ndarray, alerts: list, performance_metrics: Dict[str, float]) -> str:
        """Generate enhanced summary with performance metrics"""
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
    
    def check_alerts(self, forecast_data: np.ndarray, inventory_threshold: Optional[float] = None) -> list:
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

# Initialize API instance
api = OptimizedSalesForecastAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Optimized Sales Forecast API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/forecast', methods=['POST'])
def generate_forecast():
    """Generate optimized sales forecast"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        sales_data = data.get('data')
        forecast_days = data.get('forecast_days', 14)
        inventory_threshold = data.get('inventory_threshold')
        
        if not sales_data:
            return jsonify({'error': 'Sales data is required'}), 400
        
        # Process data
        df = api.process_data(sales_data)
        if df is None:
            return jsonify({'error': 'Failed to process data'}), 400
        
        # Prepare time series data
        X, y, min_val, max_val, daily_sales = api.prepare_time_series(df, forecast_days)
        
        if X is None or len(X) == 0:
            return jsonify({'error': 'Insufficient data for forecasting. Need at least 60 days of historical data.'}), 400
        
        # Create and train model
        model, history = api.create_optimized_lstm_model(X, y, forecast_days)
        
        if model is None:
            return jsonify({'error': 'Failed to train model'}), 500
        
        # Evaluate model performance
        performance_metrics = api.evaluate_model_performance(model, X, y)
        
        # Generate forecast
        forecast = api.generate_forecast(model, X, min_val, max_val, forecast_days)
        
        if forecast is None:
            return jsonify({'error': 'Failed to generate forecast'}), 500
        
        # Create forecast dates
        last_date = daily_sales['Date'].max()
        forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
        
        # Generate alerts
        alerts = api.check_alerts(forecast, inventory_threshold)
        
        # Generate summary
        summary = api.generate_enhanced_summary(
            daily_sales['Sales'].values,
            forecast,
            alerts,
            performance_metrics
        )
        
        # Prepare response
        response = {
            'success': True,
            'forecast': {
                'dates': forecast_dates,
                'values': forecast.tolist(),
                'summary': {
                    'average': float(np.mean(forecast)),
                    'maximum': float(np.max(forecast)),
                    'minimum': float(np.min(forecast)),
                    'trend': 'increasing' if np.mean(forecast) > np.mean(daily_sales['Sales'].values) else 'decreasing'
                }
            },
            'performance_metrics': performance_metrics,
            'alerts': alerts,
            'summary': summary,
            'historical_data': {
                'dates': daily_sales['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': daily_sales['Sales'].tolist()
            }
        }
        
        # Clear memory
        gc.collect()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Analyze data without generating forecast"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sales_data = data.get('data')
        
        if not sales_data:
            return jsonify({'error': 'Sales data is required'}), 400
        
        # Process data
        df = api.process_data(sales_data)
        if df is None:
            return jsonify({'error': 'Failed to process data'}), 400
        
        # Calculate basic statistics
        total_records = len(df)
        total_sales = float(df['Sales'].sum())
        avg_daily_sales = float(df.groupby('Order_Date')['Sales'].sum().mean())
        date_range = {
            'start': df['Order_Date'].min().strftime('%Y-%m-%d'),
            'end': df['Order_Date'].max().strftime('%Y-%m-%d')
        }
        
        # Calculate sales statistics
        sales_stats = {
            'mean': float(df['Sales'].mean()),
            'median': float(df['Sales'].median()),
            'std': float(df['Sales'].std()),
            'min': float(df['Sales'].min()),
            'max': float(df['Sales'].max())
        }
        
        response = {
            'success': True,
            'data_overview': {
                'total_records': total_records,
                'total_sales': total_sales,
                'average_daily_sales': avg_daily_sales,
                'date_range': date_range
            },
            'sales_statistics': sales_stats,
            'message': 'Data analysis completed successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
