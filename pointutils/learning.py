# -*- coding: utf-8 -*-
"""
learning module 

Description
-----------
machine learning applied to point clouds

"""

# This must go first or it causes the error:
#ImportError: dlopen: cannot load any more object with static TLS
try:
    #import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    import xgboost as xgb
except ImportError:
    pass
    print('xgb not available')

from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from osgeo import gdal, ogr#,osr
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
import joblib
from sklearn import metrics
import joblib as jb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from tpot import TPOTClassifier, TPOTRegressor
from pyntcloud import PyntCloud

from keras.models import Sequential

#from keras.models import load_model, save_model
# if still not working try:
from keras.layers.core import Dense#, Dropout, Flatten, Activation

from keras.wrappers.scikit_learn import KerasClassifier
#import tensorflow as tf
#from keras.utils import multi_gpu_model
import os
from glob2 import glob
gdal.UseExceptions()
ogr.UseExceptions()

def create_model_tpot(X_train, outModel, cv=6, cores=-1,
                      regress=False, params = None, scoring=None):
    
    """
    Create a model using the tpot library where genetic algorithms
    are used to optimise pipline and params. 

    Parameters
    ----------  
    
    X_train: np array
              numpy array of training data where the 1st column is labels
    
    outModel: string
               the output model path (which is a .py file)
               from which to run the pipeline
    
    cv: int
         no of folds
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    strat: bool
            a stratified grid search
    
    regress: bool
              a regression model if True, a classifier if False
    
    params: a dict of model params (see tpot)
             enter your own params dict rather than the range provided
    
    scoring: string
              a suitable sklearn scoring type (see notes)
                           
    """
    #t0 = time()
    
    print('Preparing data')   
    
    """
    Prep of data for model fitting 
    """

    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    if params is None and regress is False:       
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2,
                              n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is False:
        tpot = TPOTClassifier(config_dict=params, n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params is None and regress is True:       
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2,
                              n_jobs=cores, scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is True:
        tpot = TPOTRegressor(config_dict=params, n_jobs=cores, verbosity=2,
                             scoring = scoring,
                              warm_start=True)
        tpot.fit(X_train, y_train)

    tpot.export(outModel)    


def create_model(X_train, outModel, clf='svc', random=False, cv=6, cores=-1,
                 strat=True, regress=False, params = None, scoring=None, 
                 ply=False, save=True):
    
    """
    Brute force or random model creating using scikit learn. Either use the
    default params in this function or enter your own (recommended - see sklearn)
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels
    
    outModel: string
               the output model path which is a gz file, if using keras it is 
               h5 
    
    clf : string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
          
          keras nnt also available as a very limited option - the nnt is 
          currently a dense sequential of 32, 16, 8, 32 - please inspect the
          source. If using GPU, you will likely be limited to a sequential 
          grid search as multi-core overloads the GPUs quick!
          
    random: bool
             if True, a random param search
    
    cv: int
         no of folds
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    strat: bool
            a stratified grid search
    
    regress : bool
              a regression model if True, a classifier if False
    
    params : a dict of model params (see scikit learn)
             enter your own params dict rather than the range provided
    
    scoring : string
              a suitable sklearn scoring type (see notes)
    
        
    Notes:
    ------      
        Scoring types - there are a lot - some of which won't work for 
        multi-class, regression etc - see the sklearn docs!
        
        'accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
        'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
        'neg_mean_absolute_error', 'neg_mean_squared_error',
        'neg_median_absolute_error', 'precision', 'precision_macro',
        'precision_micro', 'precision_samples', 'precision_weighted',
        'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
        'recall_weighted', 'roc_auc'
                    
    """
    #t0 = time()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    print('Preparing data')   
    # TODO IMPORTANT add xgb boost functionality
    #inputImage = gdal.Open(inputIm)    
    
    """
    Prep of data for model fitting 
    """

    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    if ply == False:
        
        X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    if scoring is None and regress is False:
        scoring = 'accuracy'
    elif scoring is None and regress is True:    
        scoring = 'r2'
    # Choose the classifier type
    # TODO this has become rather messy (understatement)
    # and inefficient - need to make it more 
    # elegant
    no_classes = len(np.unique(y_train))
    if clf == 'keras':
        
        
        kf = StratifiedKFold(cv, shuffle=True)
        #Not currently workable
#        if gpu > 1:
#            
#
#
#            def _create_nnt(no_classes=no_classes):
#            	# create model - fixed at present
#                tf.compat.v1.disable_eager_execution()
#                with tf.device("/cpu:0"):
#                    model = Sequential()
#                    model.add(Dense(32, activation='relu', input_dim=bands))
#                    model.add(Dense(16, activation='relu'))
#                    model.add(Dense(8,  activation='relu'))
#                    model.add(Dense(32, activation='relu'))
#                    model.add(Dense(no_classes, activation='softmax'))
#                    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
#                                metrics=['accuracy'])
#                    model = multi_gpu_model(model, gpus=2)
#                    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
#                                metrics=['accuracy'])
#                    return model
       # else:
        def _create_nnt(no_classes=no_classes):
    	# create model - fixed at present 
        
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=bands))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(8,  activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(no_classes, activation='softmax'))
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                        metrics=['accuracy'])
            return model
        
        # the model
        model = KerasClassifier(build_fn=_create_nnt, verbose=1)
        # may require this to get both GPUs working
        
		# initialize the model

        # define the grid search parameters
        if params is None:
            batch_size = [10, 20, 40]#, 60, 80, 100]
            epochs = [10]#, 30]
            param_grid = dict(batch_size=batch_size, epochs=epochs)
        else:
            param_grid = params
        
        # It is of vital importance here that the estimator model is passed
        # like this otherwiae you get loky serialisation error
        # Also, at present it has to work sequentially, otherwise it overloads
        # the gpu
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, 
                            cv=kf, verbose=1)
        grid.fit(X_train, y_train)
        
        grid.best_estimator_.model.save(outModel)
        

        
        
    if clf == 'erf':
         RF_clf = ExtraTreesClassifier(n_jobs=cores)
         if random==True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
                      
        # run randomized search
            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                       n_jobs=-1, n_iter=20,  verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
         else:
            if params is None: 
            #currently simplified for processing speed 
                param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100],
                             "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:               
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:  
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
                
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
         
    if clf == 'xgb' and regress is False:
        xgb_clf = XGBClassifier()
        if params is None:
                # This is based on the Tianqi Chen author of xgb
                # tips for data science as a starter
                # he recommends fixing trees - they are 200 by default here
                # crunch this first then fine tune rest
                # 
                ntrees = 200
                param_grid={'n_estimators': [ntrees],
                            'learning_rate': [0.1], # fine tune last
                            'max_depth': [4, 6, 8, 10],
                            'colsample_bytree': [0.4,0.6,0.8,1.0]}
            #total available...
#            param_grid={['reg_lambda',
#                         'max_delta_step',
#                         'missing',
#                         'objective',
#                         'base_score',
#                         'max_depth':[6, 8, 10],
#                         'seed',
#                         'subsample',
#                         'gamma',
#                         'scale_pos_weight',
#                         'reg_alpha', 'learning_rate',
#                         'colsample_bylevel', 'silent',
#                         'colsample_bytree', 'nthread', 
#                         'n_estimators', 'min_child_weight']}
        else:
            param_grid = params
        grid = GridSearchCV(xgb_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
    if clf == 'gb' and regress is False:

        gb_clf = GradientBoostingClassifier()
        if params is None:
            param_grid ={"n_estimators": [100], 
                         "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                         "max_features": ['sqrt', 'log2'],                                                
                         "max_depth": [3,5],                    
                         "min_samples_leaf": [5,10,20,30]}
        else:
            param_grid = params

        if strat is True and regress is False:               
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False:
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    if clf == 'gb'  and regress is True:
        gb_clf = GradientBoostingRegressor(n_jobs=cores)
        if params is None:
            param_grid = {"n_estimators": [500],
                          "loss": ['ls', 'lad', 'huber', 'quantile'],                      
                          "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                          "max_features": ['sqrt', 'log2'],                                                
                          "max_depth": [3,5],                    
                          "min_samples_leaf": [5,10,20,30]}
        else:
            param_grid = params
        
        grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    if clf == 'rf' and regress is False:
         RF_clf = RandomForestClassifier(n_jobs=cores, random_state = 123)
         if random==True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
                      

            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                       n_jobs=-1, n_iter=20,  verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 

         else:
            if params is None: 

                param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100,200,500],
                             "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:               
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:  
                grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
                
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 
         
    if clf == 'rf' and regress is True:
        RF_clf = RandomForestRegressor(n_jobs = cores, random_state = 123)
        if params is None:
            param_grid ={"n_estimators": [500],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
                             "min_samples_split": [2,3,5],
                             "min_samples_leaf": [5,10,20,50,100,200,500],
                             "bootstrap": [True, False]}
        else:
            param_grid = params
        grid = GridSearchCV(RF_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
                
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 

    if clf == 'linsvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.LinearSVC()
        if random == True:
            param_grid = [{'C': [expon(scale=100)], 'class_weight':['auto', None]}]
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
        else:
             param_grid = [{'C': [1, 10, 100, 1000], 'class_weight':['auto', None]}]
            #param_grid = [{'kernel':['rbf', 'linear']}]
             if strat is True:               
                grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
             elif strat is False and regress is False:  
                 grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
    if clf == 'linsvc' and regress is True:
        svm_clf = svm.LinearSVR()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000]},
                           {'loss':['epsilon_insensitive',
                        'squared_epsilon_insensitive']}]
        else:
            param_grid = params
        grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    if clf == 'svc': # Far too bloody slow
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.SVC(probability=False)
        if random == True:
            if params is None:
        
                param_grid = [{'C': [expon(scale=100)], 'gamma': [expon(scale=.1).astype(float)],
                  'kernel': ['rbf'], 'class_weight':['auto', None]}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))

        if params is None:
    
             param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
             'kernel': ['rbf'], 'class_weight':['auto', None]}]
        else:
            param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:               
            grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False: 
            grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
             #print("done in %0.3fs" % (time() - t0))
    
    if clf == 'nusvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.NuSVC(probability=False)
        if random == True:
            if params is None:
                param_grid = [{'nu':[0.25, 0.5, 0.75, 1], 'gamma': [expon(scale=.1).astype(float)],
                                          'class_weight':['auto']}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel) 
            #print("done in %0.3fs" % (time() - t0))
        else:
            if params is None:
                 param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4],
                                'class_weight':['balanced']}]
            else:
                param_grid = params
            #param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:               
                grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
        elif strat is True and regress is False: 
             grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
    if clf == 'nusvc' and regress is True:
         svm_clf = svm.NuSVR()
         param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4]}]
         grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
         grid.fit(X_train, y_train)
         joblib.dump(grid.best_estimator_, outModel) 

    if clf == 'logit':
        logit_clf = LogisticRegression()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000], 'penalty': ['l1', 'l2', ],
                           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                           'multi_class':['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 
        
    if clf == 'sgd':
        logit_clf = SGDClassifier()
        if params is None:
            param_grid = [{'loss' : ['hinge, log', 'modified_huber',
                                     'squared_hinge', 'perceptron'], 
                           'penalty': ['l1', 'l2', 'elasticnet'],
                           'learning_rate':['constant', 'optimal', 'invscaling'],
                           'multi_class':['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel) 

    return [grid.best_estimator_, grid.cv_results_, grid.best_score_, grid.best_params_]
#    print(grid.best_params_)
#    print(grid.best_estimator_)
#    print(grid.oob_score_)
 

def RF_oob_opt(model, X_train, min_est, max_est, step, regress=False):
    
    """ 
    This function uses the oob score to find the best parameters.
    
    This cannot be parallelized due to the warm start bootstrapping, so is
    potentially slower than the other cross val in the create_model function
        
    This function is based on an example from the sklearn site
        
    This function plots a graph diplaying the oob rate
        
    Parameters
    ---------------------
    
    model: string (.gz)
            path to model to be saved
    
    X_train: np array
              numpy array of training data where the 1st column is labels
    
    min_est: int
              min no of trees
    
    max_est: int
              max no of trees
    
    step: int
           the step at which no of trees is increased
    
    regress: bool
              boolean where if True it is a regressor
    
    Returns: tuple of np arrays
    -----------------------
        
    error rate, best estimator
        
    """
    
    RANDOM_STATE = 123
    print('Preparing data')    
    
    """
    Prep of data for classification - getting bands one at a time to save memory
    """
    print('processing data for classification')

    
    bands = X_train.shape[1]-1
    
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if regress is False:
        X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]
    
    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    print('iterating estimators')
    if regress is True:
        max_feat = X_train.shape[1]-1
        ensemble_clfs = [
        ("RandomForestClassifier, max_features='no_features'",
                RandomForestRegressor(warm_start=True, oob_score=True,
                                       max_features=max_feat,
                                       random_state=RANDOM_STATE))]
    else:    
        ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
                RandomForestClassifier(warm_start=True, oob_score=True,
                                       max_features="sqrt",
                                       random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
                RandomForestClassifier(warm_start=True, max_features='log2',
                                       oob_score=True,
                                       random_state=RANDOM_STATE)),
        ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    
    # Range of `n_estimators` values to explore.
    min_estimators = min_est
    max_estimators = max_est
    
    for label, clf in ensemble_clfs:
        for i in tqdm(range(min_estimators, max_estimators + 1, step)):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)
    
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    
    # suspect a slow down after here, likely the plot...
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="best")
    plt.show()
    
    # regression option
    if regress is True:
        max_features = max_feat
        er = np.array(error_rate["RandomForestClassifier, max_features='no_features'"][0:max_estimators])
        bestscr = er[:,1].min()
        data = er[np.where(er[:,1] == bestscr)]
        best_param = ((max_features, data[0]))
        best_model = RandomForestRegressor(warm_start=True, oob_score=True,
                                       max_features= max_features,
                                       n_estimators=best_param[1][0].astype(int),
                                       n_jobs=-1)
    else:    
        sqrt = np.array(error_rate["RandomForestClassifier, max_features='sqrt'"][0:max_estimators])
        log2 = np.array(error_rate["RandomForestClassifier, max_features='log2'"][0:max_estimators])
    #NoFeat = np.array(error["RandomForestClassifier, max_features='None'"][0:max_estimators])
        minsqrt = sqrt[:,1].min()
        minLog = log2[:,1].min()
        
        if minsqrt>minLog:
            minVal = minLog
            max_features = 'log2'
            data = log2[np.where(log2[:,1] == minVal)]
        else:
            minVal = minsqrt
            max_features = 'sqrt'
            data = sqrt[np.where(sqrt[:,1] == minVal)]
    
    
    
        best_param = ((max_features, data[0]))
    

        best_model = RandomForestClassifier(warm_start=True, oob_score=True,
                                       max_features= max_features,
                                       n_estimators=best_param[1][0].astype(int),
                                       n_jobs=-1)
    best_model.fit(X_train, y_train)                
    joblib.dump(best_model, model) 
    return error_rate, best_param


def plot_feature_importances(modelPth, featureNames, model_type='scikit'):
    
    """
    Plot the feature importances of an ensemble classifier
    
    Parameters
    --------------------------
    
    modelPth : string
               A sklearn model path 
    
    featureNames : list of strings
                   a list of feature names
    
    """
    
    model = joblib.load(modelPth)
    
    if model_type=='scikit':
        n_features = model.n_features_
    if model_type=='xgb':
        n_features = model.n_features_in_
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), featureNames)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()


def get_training_ply(incld, label_field="training", classif_field='label',
                     rgb=True, outFile=None,  
                     ignore=['x', 'y', 'scalar_ScanAngleRank', 'scalar_NumberOfReturns',
                         'scalar_ReturnNumber', 'scalar_GpsTime','scalar_PointSourceId']):
    
    """ 
    Get training from a point cloud
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    label_field: string
              the name of the field representing the training points which must
              be positive integers
              
    classif_field: string
              the name of the field that will be used for classification later
              must be specified so it can be ignored
    rgb: bool
              whether there is rgb data to be included             
                
    outFile: string
               path to training array to be saved as .gz via joblib
    
    ignore: list
           the pointcloud attributes to ignore for training
    Returns
    -------
    
    np array of training where first column is labels
    
    list of feature names for later ref/plotting

    """  
    
    pcd = PyntCloud.from_file(incld)
    
    dF = pcd.points

    # Must be done sperately otherwise py appends to list outside of function
    if classif_field != None:
        del dF[classif_field]

    
    # If any of the above list exists, cut them from the dF

    cols = dF.columns.to_list()

    for i in ignore:
        if i in cols:
            del dF[i]
    del cols
#    pProps =['anisotropy', 'curvature', "eigenentropy", "eigen_sum",
#             "linearity", "omnivariance", "planarity", "sphericity"]
    
    # python bug this var persists/pointer or somehting
    del ignore      
   
    label = dF[label_field].to_numpy()
    
    del dF[label_field]
    
    features = dF.to_numpy()
    
    label.shape = (label.shape[0], 1)
    
    X_train = np.hstack((label, features))
    
    # retain feat names for plot potentiallu
    
    fnames = dF.columns.to_list()
    
    del features, dF, label
    
    # prep for sklearn
    X_train = X_train[X_train[:,0] >= 0]
        
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    

    if outFile != None:
        jb.dump(X_train, outFile, compress=2)
    
    return X_train, fnames

def get_training_tiles(folder, label_field="training", classif_field='label',
                     rgb=True, outFile=None,  
                     ignore=['x', 'y', 'scalar_ScanAngleRank', 'scalar_NumberOfReturns',
                         'scalar_ReturnNumber', 'scalar_GpsTime','scalar_PointSourceId']):
    
    """ 
    Get training from a point cloud
    
    
    Parameters 
    ----------- 
    
    folder: string
              the input folder containing .ply files
    
    label_field: string
              the name of the field representing the training points which must
              be positive integers
              
    classif_field: string
              the name of the field that will be used for classification later
              must be specified so it can be ignored
    rgb: bool
              whether there is rgb data to be included             
                
    outFile: string
               path to training array to be saved as .gz via joblib
    
    ignore: list
           the pointcloud attributes to ignore for training
    Returns
    -------
    
    np array of training where first column is labels
    
    list of feature names for later ref/plotting

    """  
    # Just loop the above func
    # TODO a suitable parallel option
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    trainlist = []
    
    for f in plylist:
    
        train, fnames = get_training_ply(f, label_field=label_field, 
                                           classif_field=classif_field,
                                           rgb=rgb, outFile=None,
                                           ignore=ignore)
        trainlist.append(train)
        del train
    
    X_train = np.vstack(trainlist)
    
    if outFile != None:
        jb.dump(X_train, outFile, compress=2)
    
    return X_train, fnames
    
    

def classify_ply(incld, inModel, train_field="training", class_field='label',
                 rgb=True, outcld=None,
                 ignore=['x', 'y', 'scalar_ScanAngleRank', 'scalar_NumberOfReturns',
                         'scalar_ReturnNumber', 'scalar_GpsTime','scalar_PointSourceId']):
    
    """ 
    Classify a point cloud (ply format)
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
                
    class_field: string
               the name of the field that the results will be written to
               this must already exist! Create in CldComp. or cgal
    train_field: string
              the name of the training label field so it can be ignored
    
    rgb: bool
        whether there is rgb data to be included
                 
    outcld: string
               path to a new ply to write if not writing to the input one
   
    ignore: list
           the pointcloud attributes to ignore for classification
    """  
    
    
    # TODO - I/O classify by chunk?
    
    pcd = PyntCloud.from_file(incld)
    
    dF = pcd.points
    
    # required to ensure we don't lose field later
    # bloody classes
    del pcd
    
    # Must be done sperately otherwise py appends to list outside of function
    del dF[class_field]
    del dF[train_field]

    # If any of the above list exists, cut them from the dF
    cols = dF.columns.to_list()

    for i in ignore:

        if i in cols:
            del dF[i]

    del cols
    
    # python bug this var persists/pointer or somehting
    del ignore        
    
    X = dF.to_numpy()
    
    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]
    del dF
    print('Classifying')
    
    # if a keras model
    if os.path.splitext(inModel)[1] == ".h5":
        model1 = load_model(inModel)
        predictClass = model1.predict(X)
        # get the class based on the location of highest prob
        predictClass = np.argmax(predictClass,axis=1)
    else:  
        model1 = joblib.load(inModel)
        predictClass = model1.predict(X)

    # read the files in again due to the earlier issue
    pcd = PyntCloud.from_file(incld)
    
    pcd.points[class_field] = np.int32(predictClass)
    
    if outcld == None:    
        pcd.to_file(incld)
    else:
        pcd.to_file(outcld)


def classify_ply_tile(folder, inModel, train_field="training", class_field='label',
                 rgb=True, outcld=None,
                 ignore=['x', 'y', 'scalar_ScanAngleRank', 'scalar_NumberOfReturns',
                         'scalar_ReturnNumber', 'scalar_GpsTime','scalar_PointSourceId']):
    """ 
    Classify a point cloud (ply format)
    
    
    Parameters 
    ----------- 
    
    folder: string
              the input folder containing .ply files
                
    class_field: string
               the name of the field that the results will be written to
               this must already exist! Create in CldComp. or cgal
    train_field: string
              the name of the training label field so it can be ignored
    
    rgb: bool
        whether there is rgb data to be included
                 
    outcld: string
               path to a new ply to write if not writing to the input one
   
    ignore: list
           the pointcloud attributes to ignore for classification
    """ 
            
            
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    for f in plylist:
        classify_ply(f, inModel, train_field=train_field,
                     class_field=class_field,
                     rgb=rgb, outcld=None,
                     ignore=ignore)
    
def rmse_vector_lyr(inShape, attributes):

    """ 
    Using sklearn get the rmse of 2 vector attributes 
    (the actual and predicted of course in the order ['actual', 'pred'])
    
    
    Parameters 
    ----------- 
    
    inShape: string
              the input vector of OGR type
        
    attributes: list
           a list of strings denoting the attributes
         
    """    
    
    #open the layer etc
    shp = ogr.Open(inShape)
    lyr = shp.GetLayer()
    labels = np.arange(lyr.GetFeatureCount())
    
    # empty arrays for att
    pred = np.zeros((1, lyr.GetFeatureCount()))
    true = np.zeros((1, lyr.GetFeatureCount()))
    
    for label in labels: 
        feat = lyr.GetFeature(label)
        true[:,label] = feat.GetField(attributes[0])
        pred[:,label] = feat.GetField(attributes[1])
    
    
    
    error = np.sqrt(metrics.mean_squared_error(true, pred))
    
    return error




