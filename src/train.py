# src/train.py
import sys
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Add parent directory to path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_preparation import DataPreparation
from src.model import GradePredictor

def plot_training_history(history):
    """Visualize training progress"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Model Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Model Accuracy Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to project root
    plot_path = os.path.join(project_root, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Training history plot saved to {plot_path}")
    plt.show()

def validate_model(predictor, scaler, X_test, y_test):
    """Validate model makes sensible predictions"""
    print("\n" + "="*60)
    print("MODEL VALIDATION - SANITY CHECKS")
    print("="*60)
    
    # Test cases with expected ranges
    test_cases = [
        ([1.0, 1.0], 25, 45, "Terrible student"),
        ([5.0, 75.0], 60, 80, "Average student"),
        ([10.0, 100.0], 90, 100, "Perfect student"),
    ]
    
    all_passed = True
    
    for raw_input, min_expected, max_expected, description in test_cases:
        # Scale and predict
        scaled_input = scaler.transform([raw_input])
        prediction = predictor.predict(scaled_input)[0][0]
        
        # Check if in range
        passed = min_expected <= prediction <= max_expected
        status = "✓ PASS" if passed else "❌ FAIL"
        
        print(f"\n{description}:")
        print(f"  Input: Study={raw_input[0]:.1f}h, Attendance={raw_input[1]:.1f}%")
        print(f"  Prediction: {prediction:.1f} points")
        print(f"  Expected range: {min_expected}-{max_expected} points")
        print(f"  {status}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("="*60)
        return True
    else:
        print("❌ VALIDATION FAILED - MODEL LEARNED INCORRECTLY")
        print("="*60)
        return False

def main():
    print("\n" + "="*60)
    print("STUDENT GRADE PREDICTOR - TRAINING PIPELINE")
    print("="*60)
    
    # Build paths
    data_path = os.path.join(project_root, 'data', 'student_data.csv')
    model_path = os.path.join(project_root, 'models', 'grade_predictor.h5')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n[Step 1/7] Loading and preparing data...")
    
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please run 'python data/generate_data.py' first")
        return
    
    data_prep = DataPreparation(data_path)
    X_train, X_test, y_train, y_test = data_prep.load_and_split()
    
    # Step 2: Build model
    print("\n[Step 2/7] Building model...")
    predictor = GradePredictor()
    predictor.build_model(input_dim=2)
    
    # Step 3: Train
    print("\n[Step 3/7] Training model...")
    history = predictor.train(X_train, y_train, epochs=100)
    
    # Step 4: Evaluate
    print("\n[Step 4/7] Evaluating model...")
    test_loss, test_mae = predictor.model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Mean Absolute Error: {test_mae:.2f} points")
    print(f"Mean Squared Error:  {test_loss:.2f}")
    print(f"\nInterpretation: On average, predictions are off by {test_mae:.2f} points")
    
    # Step 5: Validate model makes sense
    print("\n[Step 5/7] Validating model predictions...")
    validation_passed = validate_model(predictor, data_prep.scaler, X_test, y_test)
    
    if not validation_passed:
        print("\n❌ Model failed validation. NOT saving.")
        print("Try training again or check your data.")
        return
    
    # Step 6: Make sample predictions
    print("\n[Step 6/7] Making sample predictions...")
    sample_students = [
        [8.0, 95.0],  # High study hours, high attendance
        [2.0, 60.0],  # Low study hours, low attendance
        [5.0, 80.0],  # Medium study hours, good attendance
        [7.0, 70.0],  # High study, medium attendance
        [3.0, 90.0]   # Low study, high attendance
    ]
    
    sample_scaled = data_prep.scaler.transform(sample_students)
    predictions = predictor.predict(sample_scaled)
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    print(f"{'Study Hours':<15}{'Attendance %':<15}{'Predicted Score':<20}")
    print("-" * 60)
    for student, pred in zip(sample_students, predictions):
        print(f"{student[0]:<15.1f}{student[1]:<15.1f}{pred[0]:<20.1f}")
    
    # Step 7: Save model and scaler
    print("\n[Step 7/7] Saving model and scaler...")
    
    # Save model
    predictor.save(model_path)
    
    # Save scaler
    print(f"Saving scaler to: {scaler_path}")
    joblib.dump(data_prep.scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Verify files were saved
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Scaler file exists: {os.path.exists(scaler_path)}")
    
    if os.path.exists(model_path):
        print(f"Model size: {os.path.getsize(model_path)} bytes")
    if os.path.exists(scaler_path):
        print(f"Scaler size: {os.path.getsize(scaler_path)} bytes")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE AND VALIDATED!")
    print("="*60)
    print("\nFiles saved:")
    print(f"  - Model: {model_path}")
    print(f"  - Scaler: {scaler_path}")
    print("\nNext steps:")
    print("1. Check training_history.png to see learning curves")
    print("2. Run 'python src/predict.py' to make new predictions")

if __name__ == "__main__":
    main()