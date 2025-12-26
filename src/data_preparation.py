# src/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        
    def load_and_split(self, test_size=0.2):
        """Load data and split into train/test sets"""
        # Load CSV
        df = pd.read_csv(self.filepath)
        print(f"✓ Loaded {len(df)} records from {self.filepath}")
        
        # Show data ranges BEFORE scaling
        print("\nData Ranges BEFORE scaling:")
        print(f"  Study hours:  min={df['study_hours'].min():.2f}, max={df['study_hours'].max():.2f}, mean={df['study_hours'].mean():.2f}")
        print(f"  Attendance:   min={df['attendance_percent'].min():.2f}, max={df['attendance_percent'].max():.2f}, mean={df['attendance_percent'].mean():.2f}")
        print(f"  Final scores: min={df['final_score'].min():.2f}, max={df['final_score'].max():.2f}, mean={df['final_score'].mean():.2f}")
        
        # Features (input) and target (output)
        X = df[['study_hours', 'attendance_percent']].values
        y = df['final_score'].values
        
        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"✓ Split data: {len(X_train)} training, {len(X_test)} test samples")
        
        # Normalize features - FIT on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Features normalized")
        print(f"\nScaler learned from training data:")
        print(f"  Mean: {self.scaler.mean_}")
        print(f"  Std:  {self.scaler.scale_}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_scaler(self):
        return self.scaler