import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, List
import logging
from ..features.feature_engineering import FeatureEngineer

class DataPreprocessor:
    def __init__(self, data_dir: str = "data/raw/house_1"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        self.feature_engineer = FeatureEngineer()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all energy consumption data"""
        self.logger.info("Loading data...")
        
        # Load aggregate consumption (channel_1.dat)
        agg_data = pd.read_csv(self.data_dir / "channel_1.dat", 
                              sep=" ", 
                              header=None, 
                              names=["timestamp", "power"])
        
        # Convert timestamp
        agg_data["timestamp"] = pd.to_datetime(agg_data["timestamp"], unit="s")
        agg_data.set_index("timestamp", inplace=True)
        
        self.logger.info(f"Loaded data with shape: {agg_data.shape}")
        return agg_data
    
    def load_house_data(self):
        """
        Load and preprocess house energy consumption data
        """
        # TODO: Implement data loading logic
        pass
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data
        """
        # TODO: Implement preprocessing steps like:
        # - Handle missing values
        # - Convert timestamps
        # - Resample data to consistent intervals
        # - Create features for time of day, day of week, etc.
        pass
    
    # Resampling the data to a consistent frequency
    def resample_timeseries(self, 
                           data: pd.DataFrame, 
                           freq: str = '1H') -> pd.DataFrame:
        """
        Resample time series to specified frequency
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data
        freq : str
            Resampling frequency ('1H' for hourly, '1D' for daily, etc.)
            
        Returns:
        --------
        pd.DataFrame
            Resampled data
        """
        self.logger.info(f"Resampling data to {freq} frequency...")
        
        # Resample and compute various aggregations
        resampled = pd.DataFrame()
        resampled['power_mean'] = data['power'].resample(freq).mean()
        resampled['power_max'] = data['power'].resample(freq).max()
        resampled['power_min'] = data['power'].resample(freq).min()
        resampled['power_std'] = data['power'].resample(freq).std()
        
        self.logger.info(f"Resampled data shape: {resampled.shape}")
        return resampled

    # Handling missing values
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        self.logger.info("Handling missing values...")
        
        # Create copy
        df = data.copy()
        
        # Check missing values
        missing_stats = df.isnull().sum()
        self.logger.info(f"Missing values before handling:\n{missing_stats}")
        
        # Forward fill for short gaps
        df = df.fillna(method='ffill', limit=3)
        
        # For remaining gaps, use rolling mean
        df = df.fillna(df.rolling(window=24, center=True, min_periods=1).mean())
        
        # Any remaining missing values get filled with median
        df = df.fillna(df.median())
        
        self.logger.info(f"Missing values after handling:\n{df.isnull().sum()}")
        return df
    
    # Treating outliers
    def treat_outliers(self, 
                      data: pd.DataFrame, 
                      columns: List[str] = None,
                      method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers using specified method
        """
        self.logger.info("Treating outliers...")
        
        df = data.copy()
        if columns is None:
            if 'power_mean' in df.columns:
                columns = ['power_mean']
            else:
                columns = ['power']
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Log outlier statistics
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                self.logger.info(f"Found {len(outliers)} outliers in {col}")
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using FeatureEngineer"""
        self.logger.info("Starting feature engineering...")
        return self.feature_engineer.engineer_features(data)

    def run_preprocessing_pipeline(self, 
                                resample_freq: str = '1H',
                                treat_outliers: bool = True) -> pd.DataFrame:
        """Run the complete preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline...")
        
        # Load data
        data = self.load_data()
        
        # Resample time series
        data = self.resample_timeseries(data, freq=resample_freq)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Treat outliers if specified
        if treat_outliers:
            data = self.treat_outliers(data)
        
        self.logger.info("Preprocessing pipeline completed!")
        return data