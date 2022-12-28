import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

fig = plt.figure(figsize=(10,6))

def get_random_vals(lower_bound, upper_bound, num_vals):
    hand_measured_points = []
    model_points = []
    for i in range(num_vals):
        hand_measured_points.append(random.uniform(lower_bound, upper_bound))
        model_points.append(random.uniform(lower_bound, upper_bound))

    df = pd.DataFrame({
        'x': model_points,
        'y': hand_measured_points,
    })

    return df

df = get_random_vals(100.0, 2000.0, 20)

y = df['y']
x = df['x']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal
plt.title('Model Accuracy Residual Plot')
plt.scatter(df.x, standardized_residuals)
plt.xlabel('Predicted Values for Average Branch Length')
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()