# -*- coding: utf-8 -*-
# Cyril de Lavergne 

import os
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit
import random
import itertools

np.random.seed(1)
random.seed(1)


cv_para = 10



class LM(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('reg',LinearRegression(n_jobs=-1))])
        self.x = x
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
    def fitModel(self):
        self.clf = self.alg
        cvs = cross_val_score(self.clf,self.x, self.y, cv=cv_para)
        self.bestscore = {'mean':np.mean(cvs)/100,'std':np.std(cvs)/100}
        self.clf.fit(self.x,self.y)
        self.colnames = np.concatenate( (np.array(['Intercept']), self.x.columns.values)) 
        self.bestpara = dict(zip(self.colnames,np.concatenate((self.clf.named_steps["reg"].intercept_,self.clf.named_steps["reg"].coef_[0]))))
         
    def returnPreds(self,y):
        if self.df is Tarue:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x) 

class SUBSET(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('reg',LinearRegression(n_jobs=-1))])
        self.x = x
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
    def fitModel(self):
        self.score = - np.inf
        for k in range(1,len(self.x.columns.values)):
            for combo in itertools.combinations(self.x.columns.values, k):
                self.clf = self.alg
                score = cross_val_score(self.clf,self.x[list(combo)], self.y, cv=cv_para)
                if np.mean(score) > self.score:
                    self.clf.fit(self.x[list(combo)],self.y)
                    self.score = np.mean(score)
                    self.bestscore =  {'mean':np.mean(score)/100,'std':np.std(score)/100}
                    self.colnames = np.concatenate( (np.array(['Intercept']),list(combo))) 
                    self.bestpara = dict(zip(self.colnames,np.concatenate( (self.clf.named_steps["reg"].intercept_, self.clf.named_steps["reg"].coef_[0]))))

    def returnPreds(self,y):
        if self.df is True:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x) 


class LASSO(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('reg',Lasso())])
        self.x = x 
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
        
    def fitModel(self):
        self.parameters = {
                'reg__alpha': np.logspace(-100, 1, num=50,base=2)
                }
        self.clf = GridSearchCV(self.alg,
                           self.parameters,
                           verbose=0, 
                           scoring="neg_mean_squared_error",
                           cv=cv_para,
                           n_jobs=-1)
        self.clf.fit(self.x,self.y)
        self.bestscore = {'mean':self.clf.cv_results_['mean_test_score'][self.clf.best_index_],
                          'std':self.clf.cv_results_['std_test_score'][self.clf.best_index_]}
        print(self.bestscore)
        self.colnames = np.concatenate( (np.array(['Intercept']), self.x.columns.values)) 
        model = self.clf.best_estimator_.named_steps["reg"]
        self.bestpara = dict(zip(self.colnames,np.concatenate( (model.intercept_, model.coef_))))
        print(self.bestpara)
        
    def returnPreds(self,y):
        if self.df is True:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf.best_estimator_,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x)    

class RIDGE(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('reg',Ridge())])
        self.x = x
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
    def fitModel(self):
        self.parameters = {
                'reg__alpha': np.logspace(-100, 1, num=50,base=2)
                }
        self.clf = GridSearchCV(self.alg,
                           self.parameters,
                           verbose=0, 
                           scoring="neg_mean_squared_error",
                           cv=cv_para,
                           n_jobs=-1)
        self.clf.fit(self.x,self.y)
        self.bestscore = {'mean':self.clf.cv_results_['mean_test_score'][self.clf.best_index_],'std':self.clf.cv_results_['std_test_score'][self.clf.best_index_]}
        self.colnames = np.concatenate( (np.array(['Intercept']), self.x.columns.values)) 
        model = self.clf.best_estimator_.named_steps["reg"]
        self.bestpara = dict(zip(self.colnames,np.concatenate( (model.intercept_, model.coef_[0]))))
        
    def returnPreds(self,y):
        if self.df is True:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf.best_estimator_,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x)    




class PCR(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('pca', PCA()), ('reg',LinearRegression())]) #n_components=n_components
        self.x = x 
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
    def fitModel(self):
        self.parameters = {
                'pca__n_components': list(range(1,len(self.x.columns.values)))
                }
        self.clf = GridSearchCV(self.alg,
                           self.parameters,
                           verbose=0, 
                           scoring="neg_mean_squared_error",
                           cv=cv_para,
                           n_jobs=-1)
        self.clf.fit(self.x,self.y)
        self.bestscore = {'mean':self.clf.cv_results_['mean_test_score'][self.clf.best_index_],'std':self.clf.cv_results_['std_test_score'][self.clf.best_index_]}
        self.colnames = np.concatenate( (np.array(['Intercept']), self.x.columns.values))
        model = self.clf.best_estimator_.named_steps["reg"]
        pca = self.clf.best_estimator_.named_steps["pca"]
        Bx = model.coef_.dot(pca.components_)[0]
        #y_intercept = plsModel.y_mean_ - np.dot(plsModel.x_mean_ , plsModel.coef_)
        # Zr = pca.transform(self.x)
        # print(Zr.shape)
        # Xr = np.transpose(np.dot(Zr, pca.components_)) #[:,0:list(self.clf.best_params_.values())[0]]
        # print(Xr.shape)
        # betaXr = np.linalg.pinv(np.dot(Xr,np.transpose(Xr)).dot(np.dot(Xr,self.y)))
        # print(betaXr)  
        self.bestpara = dict(zip(self.colnames,np.concatenate( (model.intercept_,Bx))))
        

    def returnPreds(self,y):
        if self.df is True:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf.best_estimator_,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x)    


class PLS(object):
    def __init__(self,x,y,df=True,ts=True):
        self.alg = Pipeline([('scaler',StandardScaler()), ('reg', PLSRegression())]) #n_components=n_components
        self.x = x 
        self.y = y
        self.df = df
        if ts is True:
            self.cv = TimeSeriesSplit(n_splits=cv_para)
        
    def fitModel(self):
        self.parameters = {
                'reg__n_components': list(range(1,len(self.x.columns.values)))
                }
        self.clf = GridSearchCV(self.alg,
                           self.parameters,
                           verbose=0, 
                           scoring="neg_mean_squared_error",
                           cv=cv_para,
                           n_jobs=-1)
        self.clf.fit(self.x,self.y)
        self.bestscore = {'mean':self.clf.cv_results_['mean_test_score'][self.clf.best_index_],'std':self.clf.cv_results_['std_test_score'][self.clf.best_index_]}
        self.colnames = np.concatenate( (['Intercept'], self.x.columns.values))   
        plsModel = self.clf.best_estimator_.named_steps["reg"]
        y_intercept = plsModel.y_mean_ - np.dot(plsModel.x_mean_ , plsModel.coef_)
        self.bestpara = dict(zip(self.colnames,np.concatenate( ([   y_intercept],plsModel.coef_))))
        
    def returnPreds(self,y):
        if self.df is True:
            X = np.array(self.x)
            preds = []
            for train_index, test_index in self.cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                self.alg.fit(X_train,y_train)
                preds.append(self.alg.predict(X_test))
            self.predictions = np.concatenate(preds)
        else:
            self.predictions = cross_val_predict(self.clf.best_estimator_,self.x, y,cv=self.cv,n_jobs=-1)
    
    def predict(self,x):
        return self.clf.predict(x) 


def checkPerformance(model,x,y):
    preds = model.predict(x)    
    mean_squared_error(y,preds)
    predsAcc = np.where(preds > 0, 1, -1)
    yAcc = np.where(y> 0, 1, -1)
    accuracy_score(yAcc, predsAcc)



def main():
    path = 'C:/Users/cyril/Desktop/MPhil/Stat ML/'
    fileName = 'prostate.csv'
    labelName = 'lpsa'
    data = pd.read_csv(path+fileName, index_col=[0]).reset_index(drop=True)
    data = pd.get_dummies(data)
    label = np.where(labelName == data.columns)[0]
    other = np.where(labelName != data.columns)[0]
    x = data.iloc[:,other]  
    y = data.iloc[:,label]
    pls = LM(x,y)
    pls.fitModel()

if __name__ == "__main__":
    main()  