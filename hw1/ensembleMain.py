# -*- coding: utf-8 -*-
# Cyril de Lavergne


path = 'C:/Users/cyril/Desktop/MPhil/Stat ML/'
fileName = 'prostate.csv'
labelName = 'lpsa'

import pandas as pd
import numpy as np
from ensembleFunctions import *
from sklearn.model_selection import train_test_split
import inspect
import sys
import traceback

data = pd.read_csv(path+fileName, index_col=[0]).reset_index(drop=True)
data = pd.get_dummies(data)
label = np.where(labelName == data.columns)[0]
other = np.where(labelName != data.columns)[0]
x = data.iloc[:,other]	
y = data.iloc[:,label]

# Load models
classes = inspect.getmembers(sys.modules['ensembleFunctions'], lambda a: inspect.isclass(a))
mls = [a for a in classes if  a[0].isupper()]
mls = dict(mls)


# create models
models = {}
for key, model in mls.items():
    models[key] = model(x,y)

# fit models + get parameters and metrics
scoring = pd.DataFrame(columns=['mean','std'])
coefs = pd.DataFrame(columns=np.concatenate((['Intercept'],list(x.columns.values))))
for key , model in models.items():
	print('MODEL', key)
	if key == 'PCA':
		continue
	try:
	    model.fitModel()
	    s_bestscore = pd.Series(model.bestscore).rename(key)
	    s_bestpara = pd.Series(model.bestpara).rename(key)
	    scoring= scoring.append(s_bestscore)
	    coefs= coefs.append(s_bestpara)
	except Exception as e:
		print('did not work', model, e)
		traceback.print_exc(file=sys.stdout)


print(scoring.T)
print(coefs.T)