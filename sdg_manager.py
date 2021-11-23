import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# E3 classes
from sdg import SDG, sdg_inputs

#inputs ---------
inputs = sdg_inputs()
inputs.goal = 2 # 1,2,....,17

#SDG ---------
s = SDG(inputs)

s.compute() #computes the correlation and prediction
s.display() #plots multiple graphs
s.report() #reports most important outcomes
