import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

caption = "Figure 8: Standardized Residuals displaying the accuracy of cell coverage predictions. The predicted values were software outputs, while the observed values were manually inspected."

fig = plt.figure(figsize=(10,6))
plt.rcParams.update({'font.size': 22})

# Declare the data in a pandas DataFrame
df = pd.DataFrame({
	'x': [9.168372,9.076905,8.677338,10.56686,8.106872,7.767481,12.94741,14.59622,6.583223,16.61812,3.689975,1.978577,6.542304,2.342039,1.925623,4.854977,3.064147,2.055602,1.851005,4.36635,3.851246,5.519316,3.309664,3.502227,3.020821,3.555181,3.331327,4.551691,3.711638,3.129137,3.259117,2.520159,2.951017,2.900469,4.744253,5.32194,3.095439,2.753641,2.635696,2.94861,3.610543,3.018414,2.638103,3.018414,3.340956,3.186906,3.894572,6.409917,3.003972,3.160428,3.285594,4.279697,3.292815,4.395234,2.565892,3.670718,4.590203,3.016007,5.519316,4.628716,3.545553,9.498135,11.94849,11.76796,11.46468,12.42027,10.84126,10.53075,10.86051,13.9054,17.69166,11.09159,9.864003,14.6901,16.61812,10.46095,10.31652,17.46781,15.66013,19.33566,16.18245,13.81153,6.94187,11.52004,13.09664],
	'y': [12.695,7.371,9.9,15.485,10.73,11.095,16.127,17.61,23.13,23.17,4.995,2.27,8.9,2.76,2.436,3.622,3.622,1.824,2.098,5.093,4.69,8.36,4.669,4.915,3.855,3.802,4.174,6.325,3.902,4.181,4.132,3.437,2.078,2.999,4.019,5.097,3.583,3.394,3.237,3.141,3.856,4.234,3.123,3.322,4.697,3.588,5.866,7.914,3.939,3.975,3.334,3.859,3.868,5.567,4.28,4.349,6.794,3.836,6.86,6.088,9.654,13.325,17.918,19.491,15.504,18.795,13.98,15.764,17.702,20.6,22.992,16.47,19.369,22.341,20.912,17.548,18.35,20.318,15.287,19.746,20.721,16.706,7.905,16.739,12.768],
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
plt.xlabel('Predicted Values for Cell Area (%)\n\n'+caption, wrap=True)
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()
