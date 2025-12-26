# data/generate_data.py
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 student records
n_students = 1000

# FIXED: Include full range from 0 to 10 and 0 to 100
study_hours = np.random.uniform(0, 10, n_students)
attendance = np.random.uniform(0, 100, n_students)  # Changed from (50, 100)

# Create realistic final scores
# Formula: 40% study hours, 30% attendance, 30% base ability
final_score = (study_hours * 4 + 
               attendance * 0.3 + 
               np.random.normal(30, 10, n_students))
final_score = np.clip(final_score, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance_percent': attendance,
    'final_score': final_score
})

# Verify data quality
print(f"✓ Generated {n_students} student records")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nStatistics:")
print(df.describe())

# Check coverage of extreme cases
print(f"\n" + "="*60)
print("DATA COVERAGE ANALYSIS")
print("="*60)
low_study_low_attend = df[(df['study_hours'] < 2) & (df['attendance_percent'] < 20)]
high_study_high_attend = df[(df['study_hours'] > 8) & (df['attendance_percent'] > 90)]

print(f"Students with low study (<2h) AND low attendance (<20%): {len(low_study_low_attend)}")
if len(low_study_low_attend) > 0:
    print(f"  Their scores: {low_study_low_attend['final_score'].min():.1f} - {low_study_low_attend['final_score'].max():.1f}")
    print(f"  Mean: {low_study_low_attend['final_score'].mean():.1f}")

print(f"\nStudents with high study (>8h) AND high attendance (>90%): {len(high_study_high_attend)}")
if len(high_study_high_attend) > 0:
    print(f"  Their scores: {high_study_high_attend['final_score'].min():.1f} - {high_study_high_attend['final_score'].max():.1f}")
    print(f"  Mean: {high_study_high_attend['final_score'].mean():.1f}")

# Test the formula
print(f"\n" + "="*60)
print("EXPECTED SCORES FROM FORMULA")
print("="*60)
test_cases = [
    (1, 1),
    (1, 50),
    (5, 75),
    (10, 100)
]

print("Formula: score = (study*4) + (attend*0.3) + 30 (± random noise)")
for study, attend in test_cases:
    expected = (study * 4) + (attend * 0.3) + 30
    print(f"  Study={study:2}h, Attend={attend:3}% → Expected ≈ {expected:.1f} points")

# Save to CSV
df.to_csv('data/student_data.csv', index=False)
print(f"\n✓ Data saved to data/student_data.csv")