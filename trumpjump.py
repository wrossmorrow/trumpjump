
import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('trump data morrow.csv');		# read data from csv file
print(data.describe())								# standard basic stats for data

data = np.asarray( data ) 							# convert to a numpy array style data structure

exog = np.vstack( data[:,7:9] ).astype( np.float )	# make sure "exogenous" variables stored as floats
exog = np.c_[ exog , np.ones(exog.shape[0]) ]		# add column for constants

exog[:,1] = exog[:,0] * exog[:,1]					# male/female models: "female" column has obama share
exog[:,0] = exog[:,0] - exog[:,1]					# male/female models: "male" column has obama share
# print(exog)

endg = data[:,6] > 0								# "endogenous" variables (trump jump) as a boolean

logit = sm.Logit( endg , exog ).fit()				# statsmodels Logit model and fit
print(logit.params)									# printing out coefficients
print(logit.summary())								# printing out estimation summary
