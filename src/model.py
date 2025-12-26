# src/model.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class GradePredictor:
    def __init__(self):
        self.model = None
        self.history = None
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def build_model(self, input_dim=2):
        """Create a simple neural network"""
        print("\n" + "="*50)
        print("BUILDING MODEL")
        print("="*50)
        
        self.model = keras.Sequential([
            # Input layer: 2 features (study hours, attendance)
            keras.layers.Dense(16, activation='relu', input_dim=input_dim),
            
            # Hidden layer
            keras.layers.Dense(8, activation='relu'),
            
            # Output layer: 1 number (predicted score)
            keras.layers.Dense(1)
        ])
        
        # Configure the learning process
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, epochs=200, validation_split=0.2):
        """Train the model"""
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        print(f"Epochs: {epochs}")
        print(f"Validation split: {validation_split}")
        print("\nTraining in progress...")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1  # Show progress
        )
        
        print("\n✓ Training complete!")
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath='models/grade_predictor.h5'):
        """Save trained model"""
        # Convert to absolute path
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.project_root, filepath)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"\n✓ Model saved to {filepath}")
        
    def load(self, filepath='models/grade_predictor.h5'):
        """Load trained model"""
        # Convert to absolute path
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.project_root, filepath)
        
        print(f"Looking for model at: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        
        # Build the model by making a dummy prediction
        # This initializes the model's internal state
        dummy_input = np.array([[0.0, 0.0]])
        _ = self.model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model loaded from {filepath}")