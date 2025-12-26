import pandas as pd
import numpy as np

np.random.seed(42)

n_students = 1000
study_hours = np.random.uniform(0, 10, n_students)
attendance = np.random.uniform(50, 100, n_students)

final_score = (study_hours * 4 + 
               attendance * 0.3 + 
               np.random.normal(30, 10, n_students))
final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance_percent': attendance,
    'final_score': final_score
})

df.to_csv('data/student_data.csv', index=False)
print(f"Generated {n_students} student records")