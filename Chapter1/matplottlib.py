import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# genrate a sequence of number from -10 to 10 with hundred steps
x = np.linspace(-10, 10, 100)
# create a second array using sine
y = np.sin(x)
plt.show(x, y, marker="X")
