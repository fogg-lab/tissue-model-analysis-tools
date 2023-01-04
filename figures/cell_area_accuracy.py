import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

fig = plt.figure(figsize=(10,6))

# Declare the data in a pandas DataFrame
df = pd.DataFrame({
	'x': [8.034661211, 7.295703454, 4.402455169],
	'y': [9.76, 15.73, 10.70],
})

x = df['x'] # experimental data
y = df['y'] # ground truth data

x = sm.add_constant(x)

# Fit linear regression model and get residuals
model = sm.OLS(y, x).fit()
influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal

# Plot the graph
plt.title('Cell Area Accuracy Plot')
plt.scatter(df.x, standardized_residuals)
plt.xlabel('Predicted Values for Cell Area')
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()