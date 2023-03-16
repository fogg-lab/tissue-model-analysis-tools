import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

fig = plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 22})

# Data from sheet found here: https://docs.google.com/spreadsheets/d/1k34RgHDlJp-wIw-DkAz7MyrL0M9bASq3jitGeXD5BG8/edit#gid=598864832

hand_measured_points = [85.81, 81.74, 68.26, 67.96, 59.01, 100.4, 102.27, 132.61, 125.71]
model_points = [133.55, 89.9, 73.66, 66.57, 95.33, 67.91, 66.57, 95.33, 67.91]

df = pd.DataFrame({
    'x': model_points,
    'y': hand_measured_points,
})

y = df['y']
x = df['x']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal
plt.scatter(df.x, standardized_residuals)
plt.xlabel('Predicted Values for Average Branch Length (μm)')
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()