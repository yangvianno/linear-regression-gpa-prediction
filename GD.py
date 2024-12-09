import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for interation in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate. 
        h = X.dot(theta) # Calculate hypothesis (predictions)
        error = h - y
	 
        G = (1/m) * transposedX.dot(error) # Calculate gradient
        theta = np.subtract(theta, alpha * G)  # or theta = theta - alpha * gradient
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;
        h = X.dot(theta)
        error = h - y
        atmp = (1/(2*m)) * np.sum(error**2)  # Mean squared error
        print(atmp)
        arrCost.append(atmp)
        ################PLACEHOLDER4 #start##########################
    return theta, arrCost