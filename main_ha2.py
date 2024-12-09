from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradientDescent
from dataNormalization import rescaleMatrix
 

#NOTICE: Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# There are two PLACEHODERS IN THIS SCRIPT

# parameters

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves. 
ALPHAS = [0.01, 0.05, 0.1, 0.5]
MAX_ITER = 1000 # More iterations for better convergence analysis
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
def custom_normalize(data):
    normalized = np.zeros_like(data)
    for col in range(data.shape[1]):
        column = data[:, col]
        mean = np.mean(column)
        range_val = np.max(column) - np.min(column)
        normalized[:, col] = (column - mean) / range_val
    return normalized

# Normalize data using custom implementation
sat = custom_normalize(sat)
################PLACEHOLDER2 #end##########################

 
# training data;
satTrain = sat[0:60, :]
# testing data; 
satTest = sat[60:len(sat),:]

plt.figure(figsize=(12, 8)) # Create figure for multiple learning rate comparison

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
for ALPHA in ALPHAS:
    theta = np.zeros(3) 

    xValues = np.ones((60, 3)) 
    xValues[:, 1:3] = satTrain[:, 0:2]
    yValues = satTrain[:, 2]
    # call the GD algorithm, placeholders in the function gradientDescent()
    [theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

 
#visualize the convergence curve
    plt.plot(range(0,len(arrCost)),arrCost, label=f'α = {ALPHA}')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.legend()
plt.grid(True)
plt.show()

# # Use the best learning rate for final evaluation
# ALPHA = 0.1  # Choose the best performing learning rate
# theta = np.zeros(3)
# xValues = np.ones((60, 3))
# xValues[:, 1:3] = satTrain[:, 0:2]
# yValues = satTrain[:, 2]
# theta, _ = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

#% step-3: testing
testXValues = np.ones((len(satTest), 3)) 
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
 

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
