import pandas
import matplotlib.pyplot as plt
import numpy as np

# Set the figure size
plt.rcParams["figure.figsize"] = [5.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Array for horizontal bar's position
ind = np.array([0, 1, 2, 3])

# Bar's width
width = 0.2

fig, ax = plt.subplots()

# Horizontal bar plot
ax.barh(ind, np.array([0.41, 0.38, 0.51, 0.55]), width, color='orange', label='Single Task Training')
ax.barh(ind + width, np.array([0.45, 0.48, 0.54, 0.56]), width, color='blue', label='Multi Task Training')

# Set Y-axis ticks and ticklabels
ax.set(yticks=ind + width, yticklabels=np.array(['Hghl', 'Hghl+M', 'SD', 'SD+M']),
ylim=[2*width - 1, len(ind)])

# Legend at the upper right corner
ax.legend(loc='upper right')

# Display the plot
plt.show()