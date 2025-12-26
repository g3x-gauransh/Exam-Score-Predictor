# debug_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("COMPREHENSIVE DEBUGGING")
print("="*60)

# 1. Load and inspect raw data
print("\n[1] RAW DATA INSPECTION")
print("-"*60)
df = pd.read_csv('data/student_data.csv')
print(f"Total records: {len(df)}")
print(f"\nData ranges:")
print(f"  Study hours:  {df['study_hours'].min():.2f} - {df['study_hours'].max():.2f}")
print(f"  Attendance:   {df['attendance_percent'].min():.2f} - {df['attendance_percent'].max():.2f}")
print(f"  Final scores: {df['final_score'].min():.2f} - {df['final_score'].max():.2f}")

# 2. Check for anomalies in low-performing students
print("\n[2] LOW-PERFORMING STUDENTS (study<2, attendance<60)")
print("-"*60)
low_performers = df[(df['study_hours'] < 2) & (df['attendance_percent'] < 60)]
print(f"Count: {len(low_performers)}")
if len(low_performers) > 0:
    print(f"Score range: {low_performers['final_score'].min():.1f} - {low_performers['final_score'].max():.1f}")
    print(f"Mean score: {low_performers['final_score'].mean():.1f}")
    print("\nSample low performers:")
    print(low_performers.head(10))

# 3. Visualize the relationship
print("\n[3] VISUALIZING DATA RELATIONSHIPS")
print("-"*60)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Study hours vs score
axes[0].scatter(df['study_hours'], df['final_score'], alpha=0.3)
axes[0].axvline(x=1, color='r', linestyle='--', label='Test case: 1 hour')
axes[0].axhline(y=35, color='r', linestyle='--', alpha=0.5, label='Expected ~35')
axes[0].set_xlabel('Study Hours')
axes[0].set_ylabel('Final Score')
axes[0].set_title('Study Hours vs Final Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Attendance vs score
axes[1].scatter(df['attendance_percent'], df['final_score'], alpha=0.3)
axes[1].axvline(x=1, color='r', linestyle='--', label='Test case: 1%')
axes[1].axhline(y=35, color='r', linestyle='--', alpha=0.5, label='Expected ~35')
axes[1].set_xlabel('Attendance %')
axes[1].set_ylabel('Final Score')
axes[1].set_title('Attendance vs Final Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_inspection.png', dpi=150)
print("✓ Saved visualization to data_inspection.png")
plt.show()

# 4. Test the formula directly
print("\n[4] FORMULA VALIDATION")
print("-"*60)
test_cases = [
    (1, 1),
    (1, 50),
    (5, 75),
    (10, 100)
]

print("Expected scores using formula: score = (study*4) + (attend*0.3) + 30")
for study, attend in test_cases:
    expected = (study * 4) + (attend * 0.3) + 30
    print(f"  Study={study:2}, Attend={attend:3} → {expected:.1f}")

# 5. Check what's actually in the data for our test case
print("\n[5] ACTUAL DATA NEAR TEST CASE (study≈1, attend≈1)")
print("-"*60)
near_test = df[(df['study_hours'] < 1.5) & (df['attendance_percent'] < 55)]
if len(near_test) > 0:
    print(f"Found {len(near_test)} students with study<1.5, attend<55")
    print(f"Their scores: {near_test['final_score'].min():.1f} - {near_test['final_score'].max():.1f}")
    print(f"Mean: {near_test['final_score'].mean():.1f}")
else:
    print("⚠ WARNING: NO students in dataset with very low study/attendance!")
    print("This might explain why model doesn't learn this region well")

# 6. Simulate training process
print("\n[6] SIMULATING SCALING")
print("-"*60)
X = df[['study_hours', 'attendance_percent']].values
y = df['final_score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaler learned:")
print(f"  Mean: {scaler.mean_}")
print(f"  Std:  {scaler.scale_}")

# Scale our test case
test_input = np.array([[1.0, 1.0]])
test_scaled = scaler.transform(test_input)
print(f"\nTest case scaling:")
print(f"  Original: {test_input[0]}")
print(f"  Scaled:   {test_scaled[0]}")

# Find actual similar examples in training data
print("\n[7] SIMILAR EXAMPLES IN TRAINING DATA")
print("-"*60)
# Find scaled training examples near our scaled test case
distances = np.sqrt(((X_train_scaled - test_scaled)**2).sum(axis=1))
nearest_idx = np.argsort(distances)[:5]

print("5 nearest training examples to our test case (study=1, attend=1):")
for i, idx in enumerate(nearest_idx):
    orig = X_train[idx]
    score = y_train[idx]
    print(f"  {i+1}. Study={orig[0]:.2f}, Attend={orig[1]:.2f} → Score={score:.1f}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE - Check data_inspection.png")
print("="*60)