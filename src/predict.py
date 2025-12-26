# src/predict.py
import sys
import os
import joblib
import numpy as np

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model import GradePredictor

def get_validated_input(prompt, min_val, max_val, input_type="float"):
    """
    Get and validate user input within specified range
    """
    while True:
        try:
            user_input = input(prompt)
            
            # Allow user to quit
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nExiting...")
                sys.exit(0)
            
            # Convert to appropriate type
            if input_type == "float":
                value = float(user_input)
            elif input_type == "int":
                value = int(user_input)
            else:
                value = float(user_input)
            
            # Validate range
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Invalid input! Please enter a value between {min_val} and {max_val}")
                print(f"   You entered: {value}")
                
        except ValueError:
            print(f"❌ Invalid input! Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

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
        
        # Verify scaler was loaded
        print(f"✓ Model and scaler loaded successfully")
        
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
    print("(Type 'q' to quit at any time)")
    print("-"*60)
    
    try:
        # Get validated input with proper ranges
        print("\n📚 Study Hours:")
        print("   Valid range: 0-10 hours per day")
        print("   Examples: 0=no study, 5=moderate, 10=intensive")
        study_hours = get_validated_input(
            "   Enter study hours per day (0-10): ",
            min_val=0,
            max_val=10,
            input_type="float"
        )
        
        print("\n📊 Attendance:")
        print("   Valid range: 0-100 percent")
        print("   Examples: 0=never attends, 50=half the time, 100=perfect")
        attendance = get_validated_input(
            "   Enter attendance percentage (0-100): ",
            min_val=0,
            max_val=100,
            input_type="float"
        )
        
        # Additional reasonableness check
        if study_hours > 10:
            print(f"\n⚠ WARNING: {study_hours} hours per day seems unrealistic")
            print("   There are only 24 hours in a day!")
            confirm = input("   Continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                print("Please enter a realistic value.")
                predict_new_student()
                return
        
        # Prepare data as 2D array
        student_data = np.array([[study_hours, attendance]])
        
        print(f"\n📊 Processing prediction...")
        print(f"   Input: study_hours={study_hours:.1f}, attendance={attendance:.1f}%")
        
        # Scale the data using the loaded scaler
        student_scaled = scaler.transform(student_data)
        
        # Make prediction on SCALED data
        prediction = predictor.predict(student_scaled)
        predicted_score = prediction[0][0]
        
        # Clip prediction to valid range (0-100) as safety measure
        predicted_score = np.clip(predicted_score, 0, 100)
        
        # Display result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nStudent Profile:")
        print(f"  Study Hours:  {study_hours:.1f} hours/day")
        print(f"  Attendance:   {attendance:.1f}%")
        print(f"\n  📈 Predicted Final Score: {predicted_score:.1f} points")
        
        # Confidence warning for extrapolation
        # Model was trained on 0-10 hours and 0-100%, warn if near edges
        if study_hours < 0.5 or study_hours > 9.5 or attendance < 5 or attendance > 95:
            print(f"\n  ⚠ Note: Prediction is near the edge of training data.")
            print(f"     Confidence may be lower for extreme values.")
        
        # Sanity check - warn if prediction seems inconsistent
        expected_approx = (study_hours * 4) + (attendance * 0.3) + 30
        difference = abs(predicted_score - expected_approx)
        
        if difference > 20:
            print(f"\n  ⚠ Warning: Prediction differs significantly from expected formula")
            print(f"     Expected (approximate): {expected_approx:.1f} points")
            print(f"     This may indicate the input is outside normal ranges.")
        
        # Provide context based on score
        if predicted_score >= 90:
            grade = "A"
            feedback = "Excellent performance expected! 🌟"
        elif predicted_score >= 80:
            grade = "B"
            feedback = "Good performance expected. 👍"
        elif predicted_score >= 70:
            grade = "C"
            feedback = "Satisfactory performance expected. ✓"
        elif predicted_score >= 60:
            grade = "D"
            feedback = "Passing, but could improve. 📚"
        else:
            grade = "F"
            feedback = "At risk of failing. Needs significant improvement. ⚠️"
        
        print(f"\n  Letter Grade: {grade}")
        print(f"  Feedback: {feedback}")
        
        # Provide recommendations based on input
        print(f"\n" + "-"*60)
        print("💡 RECOMMENDATIONS:")
        print("-"*60)
        
        if study_hours < 3:
            print("  📖 Consider increasing study time to at least 3-4 hours/day")
        if attendance < 70:
            print("  📅 Improve attendance - aim for at least 80%")
        if study_hours >= 7 and attendance >= 85:
            print("  ✨ Excellent habits! Keep up the great work!")
        else:
            print("  🎯 Focus on consistent study and attendance for better scores.")
        
        # Ask if user wants to try again
        print("\n" + "-"*60)
        again = input("Predict for another student? (y/n): ")
        if again.lower() == 'y':
            predict_new_student()
        else:
            print("\n" + "="*60)
            print("Thank you for using Grade Predictor!")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_new_student()