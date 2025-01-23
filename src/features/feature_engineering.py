import pandas as pd
import numpy as np
import logging
from typing import Dict, Union

class FeatureEngineer:
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        self.logger.info("Creating time-based features...")
        
        df = data.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Day period feature
        df['day_period'] = pd.cut(df['hour'],
                                bins=[-np.inf, 6, 12, 18, np.inf],
                                labels=['night', 'morning', 'afternoon', 'evening'])
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, 
                              base_col: str = 'power_mean') -> pd.DataFrame:
        """Create rolling statistics features"""
        self.logger.info("Creating rolling features...")
        
        df = data.copy()
        
        # Add rolling statistics (24-hour window)
        df['rolling_mean_24h'] = df[base_col].rolling(window=24, center=True).mean()
        df['rolling_std_24h'] = df[base_col].rolling(window=24, center=True).std()
        
        return df
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        self.logger.info("Starting feature engineering pipeline...")
        
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Add time-based features
        df = self.create_time_features(df)
        
        # Add rolling features
        base_col = 'power_mean' if 'power_mean' in df.columns else 'power'
        df = self.create_rolling_features(df, base_col=base_col)
        
        self.logger.info(f"Engineered features shape: {df.shape}")
        return df
