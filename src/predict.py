# src/predict.py
import sys
import os
import joblib
import numpy as np

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model import GradePredictor

def predict_new_student():
    """Interactive prediction for new students"""
    print("\n" + "="*60)
    print("STUDENT GRADE PREDICTOR - INFERENCE")
    print("="*60)
    
    # Build absolute paths
    model_path = os.path.join(project_root, 'models', 'grade_predictor.h5')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    
    # Load model and scaler
    print("\nLoading trained model and scaler...")
    
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        
        predictor = GradePredictor()
        predictor.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # CRITICAL: Verify scaler was loaded
        print(f"✓ Scaler loaded successfully")
        print(f"  Scaler mean (study_hours, attendance): {scaler.mean_}")
        print(f"  Scaler std  (study_hours, attendance): {scaler.scale_}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease run 'python src/train.py' first to train the model.")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*60)
    print("Enter student information:")
    print("-"*60)
    
    try:
        # Get user input
        study_hours = float(input("Study hours per day (0-10): "))
        attendance = float(input("Attendance percentage (0-100): "))
        
        # Validate input
        if not (0 <= study_hours <= 10):
            print("⚠ Warning: Study hours should be between 0-10")
        if not (0 <= attendance <= 100):
            print("⚠ Warning: Attendance should be between 0-100")
        
        # CRITICAL: Prepare data as 2D array
        student_data = np.array([[study_hours, attendance]])
        print(f"\n📊 Processing prediction...")
        print(f"  Original input: study_hours={study_hours:.1f}, attendance={attendance:.1f}")
        
        # CRITICAL: Scale the data using the loaded scaler
        student_scaled = scaler.transform(student_data)
        print(f"  After scaling:  {student_scaled[0]}")
        
        # Make prediction on SCALED data
        prediction = predictor.predict(student_scaled)
        predicted_score = prediction[0][0]
        
        # Display result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nStudent Profile:")
        print(f"  Study Hours:  {study_hours:.1f} hours/day")
        print(f"  Attendance:   {attendance:.1f}%")
        print(f"\n  Predicted Final Score: {predicted_score:.1f} points")
        
        # Sanity check - warn if prediction seems wrong
        if study_hours <= 2 and attendance <= 60 and predicted_score > 60:
            print("\n⚠ WARNING: High prediction for low study/attendance - model may need retraining!")
        
        if study_hours >= 8 and attendance >= 90 and predicted_score < 70:
            print("\n⚠ WARNING: Low prediction for high study/attendance - model may need retraining!")
        
        # Provide context
        if predicted_score >= 90:
            grade = "A"
            feedback = "Excellent performance expected!"
        elif predicted_score >= 80:
            grade = "B"
            feedback = "Good performance expected."
        elif predicted_score >= 70:
            grade = "C"
            feedback = "Satisfactory performance expected."
        elif predicted_score >= 60:
            grade = "D"
            feedback = "Passing, but could improve."
        else:
            grade = "F"
            feedback = "At risk of failing. Needs significant improvement."
        
        print(f"  Letter Grade: {grade}")
        print(f"  Feedback: {feedback}")
        
        # Expected score calculation (for debugging)
        expected_approx = (study_hours * 4) + (attendance * 0.3) + 30
        print(f"\n💡 Note: Expected score based on formula ≈ {expected_approx:.1f}")
        
        # Ask if user wants to try again
        print("\n" + "-"*60)
        again = input("Predict for another student? (y/n): ")
        if again.lower() == 'y':
            predict_new_student()
        else:
            print("\nThank you for using Grade Predictor!")
            
    except ValueError:
        print("\n❌ Error: Please enter valid numbers")
        predict_new_student()  # Try again
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_new_student()