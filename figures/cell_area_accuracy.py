import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

fig = plt.figure(figsize=(10,6))
plt.rcParams.update({'font.size': 22})

# Declare the data in a pandas DataFrame
df = pd.DataFrame({
	'x': [7.165724, 8.593092, 9.880852, 8.188711],
	'y': [9.652, 9.074, 8.336, 9.067],
})



x = df['x'] # experimental data
y = df['y'] # ground truth data

x = sm.add_constant(x)

# Fit linear regression model and get residuals
model = sm.OLS(y, x).fit()
influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal

# Plot the graph
plt.scatter(df.x, standardized_residuals)
plt.xlabel('Predicted Values for Cell Area')
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()