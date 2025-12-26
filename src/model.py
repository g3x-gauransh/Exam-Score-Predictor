# src/model.py - Alternative version
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class GradePredictor:
    def __init__(self):
        self.model = None
        self.history = None
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def build_model(self, input_dim=2):
        """Create a simple neural network"""
        print("\n" + "="*50)
        print("BUILDING MODEL")
        print("="*50)
        
        # Simpler architecture - sometimes less is more
        self.model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.2),  # Regularization
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)  # Linear output
        ])
        
        # Use custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
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
        
        # Add early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        print("\nTraining in progress...")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        print("\n✓ Training complete!")
        return self.history
    
    def predict(self, X):
        """Make predictions and clip to valid range"""
        predictions = self.model.predict(X, verbose=0)
        # Clip predictions to 0-100 range
        return np.clip(predictions, 0, 100)
    
    def save(self, filepath='models/grade_predictor.h5'):
        """Save trained model"""
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.project_root, filepath)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"\n✓ Model saved to {filepath}")
        
    def load(self, filepath='models/grade_predictor.h5'):
        """Load trained model"""
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.project_root, filepath)
        
        print(f"Looking for model at: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        
        # Build the model by making a dummy prediction
        dummy_input = np.array([[0.0, 0.0]])
        _ = self.model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model loaded from {filepath}")