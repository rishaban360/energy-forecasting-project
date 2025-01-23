import pandas as pd
import numpy as np
from pathlib import Path

class DataQualityChecker:
    def __init__(self, data_path: Path):
        self.data_path = data_path


    def check_sampling_frequency(self, df: pd.DataFrame) -> dict:
        """Verify if data maintains 6-second intervals"""
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(seconds=6)
        
        return {
            'maintains_frequency': (time_diffs == expected_diff).all(),
            'unique_frequencies': time_diffs.value_counts(),
            'gaps': time_diffs[time_diffs > expected_diff]
        }
    
    def check_value_ranges(self, df: pd.DataFrame) -> dict:
        """Check for physically impossible values"""
        return {
            'negative_values': (df < 0).sum(),
            'extreme_values': (df > df.mean() + 3*df.std()).sum(),
            'zero_values': (df == 0).sum()
        }
    
    def check_sensor_failures(self, df: pd.DataFrame) -> dict:
        """Detect potential sensor failures"""
        # Constant values for extended periods
        rolling_std = df.rolling(window=60).std()  # 5 minutes
        potential_failures = (rolling_std == 0).sum()
        
        return {
            'constant_periods': potential_failures,
            'sudden_jumps': (df.diff().abs() > df.std()*5).sum()
        }

    def generate_report(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive quality report"""
        return {
            'sampling': self.check_sampling_frequency(df),
            'values': self.check_value_ranges(df),
            'sensors': self.check_sensor_failures(df),
            'basic_stats': df.describe()
        } 