# tests/test_model.py
import unittest
import numpy as np
from src import GradePredictor

class TestGradePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = GradePredictor()
        self.predictor.build_model(input_dim=2)
        
    def test_model_creation(self):
        """Test if model is created with correct architecture"""
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(len(self.predictor.model.layers), 3)
        
    def test_prediction_shape(self):
        """Test if predictions have correct shape"""
        X_sample = np.array([[5.0, 80.0]])
        pred = self.predictor.predict(X_sample)
        self.assertEqual(pred.shape, (1, 1))
        
    def test_prediction_range(self):
        """Test if predictions are in reasonable range"""
        X_train = np.random.rand(100, 2)
        y_train = np.random.rand(100) * 100
        
        self.predictor.train(X_train, y_train, epochs=10)
        
        X_test = np.array([[0.5, 0.8]])
        pred = self.predictor.predict(X_test)
        
        # Prediction should be reasonable
        self.assertGreaterEqual(pred[0][0], -50)
        self.assertLessEqual(pred[0][0], 150)

if __name__ == '__main__':
    unittest.main()