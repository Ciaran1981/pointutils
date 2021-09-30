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
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from osgeo import gdal, ogr#,osr
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
import joblib
from sklearn import metrics
import sklearn.gaussian_process as gp
import joblib as jb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from tpot import TPOTClassifier, TPOTRegressor
from pyntcloud import PyntCloud
import pandas as pd
from joblib import Parallel, delayed
from keras.models import Sequential
# if still not working try:
from keras.layers.core import Dense#, Dropout, Flatten, Activation

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model, save_model
#import tensorflow as tf
#from keras.utils import multi_gpu_model
import os
from glob2 import glob

from pointutils.props import cgal_features_mem, std_features
gdal.UseExceptions()
ogr.UseExceptions()

from autosklearn.classification import AutoSklearnClassifier
from psutil import virtual_memory
import pointutils.handyplots as hp
import os
import sys
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Classification import *

gdal.UseExceptions()
ogr.UseExceptions()

def create_model_tpot(X_train, outModel, gen=5, popsize=50,  
                      cv=5, cores=-1, dask=False, test_size=0.2,
                      regress=False, params=None, scoring=None, verbosity=2, 
                      warm_start=False):
    
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
    
    regress: bool
              a regression model if True, a classifier if False
    
    test_size: float
                size of test set held out
    
    params: a dict of model params (see tpot)
             enter your own params dict rather than the range provided
             e.g. 
             {'sklearn.ensemble.RandomForestClassifier': {"n_estimators": [200],
                             "max_features": ['sqrt', 'log2'],                                                
                             "max_depth": [10, None],
    },

    'xgboost.sklearn.XGBClassifier': {
        'n_estimators': [200],
                            'learning_rate': [0.1, 0.2, 0.4]
    }}
             
    
    scoring: string
              a suitable sklearn scoring type (see notes)
              
    warm_start: bool
                use the previous population, useful if interactive
                           
    """

    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    
    X_train = X_train[X_train[:,0] != 0]
    

#   # params could be something like
#    params = {'sklearn.ensemble.RandomForestClassifier': {"n_estimators": [200],
#                             "max_features": ['sqrt', 'log2'],                                                
#                             "max_depth": [10, None],
#    },
#
#    'xgboost.sklearn.XGBClassifier': {
#        'n_estimators': [200],
#                            'learning_rate': [0.1, 0.2, 0.4]
#    }}

    
     
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #train/test split....
    X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    
    
    if params is None and regress is False:       
        tpot = TPOTClassifier(generations=gen, population_size=popsize, 
                              verbosity=verbosity,
                              n_jobs=cores, scoring=scoring, use_dask=dask,
                              warm_start=warm_start, memory='auto')
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is False:
        tpot = TPOTClassifier(generations=gen, population_size=popsize,
                              n_jobs=cores,  verbosity=verbosity,
                              scoring=scoring,
                              use_dask=dask, 
                              config_dict=params, 
                              warm_start=warm_start, memory='auto')
        tpot.fit(X_train, y_train)
        
    elif params is None and regress is True:       
        tpot = TPOTRegressor(generations=gen, population_size=popsize, 
                             verbosity=verbosity,
                             n_jobs=cores, scoring=scoring,
                             use_dask=dask, warm_start=warm_start, memory='auto')
        tpot.fit(X_train, y_train)
        
    elif params != None and regress is True:
        tpot = TPOTRegressor(generations=gen, population_size=popsize,
                             config_dict=params, n_jobs=cores, 
                             verbosity=verbosity,
                             scoring=scoring,
                             use_dask=dask, 
                             warm_start=warm_start, memory='auto')
        tpot.fit(X_train, y_train)

    tpot.export(outModel)
    
    

#    testresult = grid.best_estimator_.predict(X_test)
#    
#    crDf = hp.plot_classif_report(y_test, testresult, save=outModel[:-3]+'.png')
#    
#    OR
#    
#    tpot.score( X_test, y_test)

    #TODO
    # how'd we export the pipline as an object then simply load to predict?
    # interim is to return the object for now
    #joblib.dump(tpot, 
    
    #this'll do for now
    scr = tpot.score(X_test, y_test)
    print(scr)
    
    return tpot, scr

def create_model_autosk(X_train, outModel,  cores=-1, class_names=None,
                        incld_est=None,
                        excld_est=None, incld_prep=None, excld_prep=None, 
                        total_time=120, res_args={'cv':5},
                        mem_limit=None,
                        per_run=None, test_size=0.3, 
                        wrkfolder=None,
                        scoring=None, save=True, ply=False):
    
    """
    Auto-sklearn to create a model
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels
    
    outModel: string
               the output model path which is a gz file, if using keras it is 
               h5 
    
    cores: int or -1 (default)
            the no of parallel jobs
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    incld_est: list of strings
                estimators to included eg ['random_forest']
    
    excld_est: list of strings
                estimators to excluded eg ['random_forest']
    
    incld_prep: list of strings
                preproc to include
                
    excld_prep: list of strings
                preproc to include
    
    total_time: int
                time in seconds for the whole search process
    
    res_args: dict
                strategy for overfit avoidance e.g. {'cv':5}
    
    mem_limit: int
                memory limit per job
    
    per_run: int
                time limit per run
    
    test_size: float
            percentage to hold out to test
    
    wrkfolder: string
                path to dir for intermediate working
    
    scoring : string
              a suitable sklearn scoring type (see notes)
    
    
    
    Returns
    -------
    A list of:
        
    [model, classif_report]
    
    """
    # default limit set by autosk is low and prone to error, hence estimate
    # here based on mem + cores used 
    # (though using everything on offer as limit!)
    if mem_limit == None:
        #Work out RAM and divide among threads
        mem = virtual_memory()
        #ingb = mem.total / (1024.**3)
        inmb = mem.total / (1024.**2)
        mem_limit = inmb / cores
    
    
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

    # then introduce the test at the end 
    X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    
    #no_classes = len(np.unique(y_train))
    
    # seemingly this must guard the code below (which should then be indented),
    # though I don't understand this completely
    #if __name__ == "__main__"
    
    automl = AutoSklearnClassifier(time_left_for_this_task=total_time, 
                                   resampling_strategy_arguments=res_args,
                                   include_estimators=incld_est, 
                                   exclude_estimators=excld_est,
                                   include_preprocessors=incld_prep, 
                                   exclude_preprocessors=excld_prep,
                                   per_run_time_limit=per_run, 
                                   tmp_folder = wrkfolder,
                                   # per job seemingly
                                   memory_limit=mem_limit,
                                   n_jobs=cores)
    
    automl.fit(X_train, y_train)
    
    testresult = automl.predict(X_test)
    
    print(automl.leaderboard())
    # print results
    print(automl.sprint_statistics())
    
    #for ref
#    automl.cv_results_
#    automl.show_models()
    
    #save it
    joblib.dump(automl, outModel)
    
    crDf = hp.plot_classif_report(y_test, testresult, target_names=class_names,
                                  save=outModel[:-3]+'.png')
    
    plt_confmat(trueVals, predVals, cmap = plt.cm.gray, fmt="%d")

    return [automl, crDf]

    


def create_model(X_train, outModel, clf='svc', random=False, cv=5, cores=-1,
                 strat=True, test_size=0.3, regress=False, params = None,
                 scoring=None, class_names=None,
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
    
    clf: string
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
    
    test_size: float
            percentage to hold out to test
    
    regress: bool
              a regression model if True, a classifier if False
    
    params: a dict of model params (see scikit learn)
             enter your own params dict rather than the range provided
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    A list of:
        
    [grid.best_estimator_, grid.cv_results_, grid.best_score_, 
            grid.best_params_, classification_report)]
    
        
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
    # then introduce the test at the end 
    X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    
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
                                       n_jobs=cores, n_iter=20,  verbose=2)
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
                

         
    if clf == 'xgb' and regress is False:
        xgb_clf = XGBClassifier(use_label_encoder=False)
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

        
    if clf == 'gb' and regress is False:
        # Key parameter here is max depth
        gb_clf = GradientBoostingClassifier()
        if params is None:
            param_grid ={"n_estimators": [100], 
                         "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                         "max_features": ['sqrt', 'log2'],                                                
                         "max_depth": [3,5],                    
                         "min_samples_leaf": [5,10,20,30]}
        else:
            param_grid = params
#                       cut due to time
        if strat is True and regress is False:               
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False:
            grid = GridSearchCV(gb_clf, param_grid=param_grid, 
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        
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
        
    #Find best params----------------------------------------------------------
    if clf == 'rf' and regress is False:
         RF_clf = RandomForestClassifier(n_jobs=cores, random_state = 123)
         if random==True:
             if params is None:
                 param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
             else:
                  param_grid = params
                      
        # run randomized search
             grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                       n_jobs=cores, n_iter=20,  verbose=2)
            #print("done in %0.3fs" % (time() - t0))
         else:
            if params is None: 
            #currently simplified for processing speed 
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
                
            #print("done in %0.3fs" % (time() - t0))
    
    # Random can be quicker and more often than not produces close to
    # exaustive results
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
        
    if clf == 'nusvc' and regress is True:
         svm_clf = svm.NuSVR()
         param_grid = [{'nu':[0.25, 0.5, 0.75, 1],'gamma': [1e-3, 1e-4]}]
         grid = GridSearchCV(svm_clf, param_grid=param_grid, 
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
             #print("done in %0.3fs" % (time() - t0))
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
    
    if clf == 'keras':
        
        grid.best_estimator_.model.save(outModel)
    else:
    
        joblib.dump(grid.best_estimator_, outModel) 
    
    testresult = grid.best_estimator_.predict(X_test)
    
    crDf = hp.plot_classif_report(y_test, testresult, target_names=class_names,
                                  save=outModel[:-3]+'._classif_report.png')
    
    confmat = hp.plt_confmat(X_test, y_test, grid.best_estimator_, 
                             class_names=class_names, 
                   cmap=plt.cm.Blues, 
                fmt="%d", save=outModel[:-3]+'_confmat.png')
    
    return [grid.best_estimator_, grid.cv_results_, grid.best_score_, 
            grid.best_params_, crDf, confmat]
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

def get_training_ply(incld, label_field="training", feattype='cgal',
                     rgb=True, outFile=None, k=5, add_fields=None):
    
    """ 
    Get training from a point cloud on the fly via either cgal or pyntcloud
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    label_field: string
              the name of the field representing the training points which must
              be positive integers
    
    feattype: string
                either cgal or std
              
    rgb: bool
              whether there is rgb data to be included           
                
    outFile: string
               path to training array to be saved as .gz via joblib
    
    k: int or list of ints
        the number of scales at which to calculate features
        if cgal features an int eg 5
        if std features a list of ints eg [20, 40, 60]
    
    add_fields: list of strings
        additional fields to be included as features
        e.g. for a LiDAR file ['Intensity', 'NumberOfReturns']
    
    Returns
    -------
    
    np array of training where first column is labels
    
    list of feature names for later ref/plotting

    """  
    # feats
    if feattype == 'cgal':
        featdf = cgal_features_mem(incld,  k=k, rgb=rgb, parallel=True)
    else:
        featdf = std_features(incld, outcld=None, k=k,
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None, tofile=False)
        pass
    
    # labels
    pcd = PyntCloud.from_file(incld)
    
    labels = pcd.points[label_field].to_numpy()
    
    labels.shape = (labels.shape[0], 1)
    
    # think z may have been no good...
    # reinstate???
    #featdf['z'] = pcd.points['z'].to_numpy()
    
    if rgb == True:
        featdf['red'] = pcd.points['red'].to_numpy()
        featdf['green'] = pcd.points['green'].to_numpy()
        featdf['blue'] = pcd.points['blue'].to_numpy()
    
    if add_fields != None:
        # could be adding return no, z, intensity etc if it is LiDAR
        for a in add_fields:
            featdf[a] = pcd.points[a].to_numpy()
        
    X_train = np.hstack((labels, featdf.to_numpy()))

    # retain feat names for plot potential and later reminders
    fnames = list(featdf.columns)
    
    del featdf, labels
    
    # prep for sklearn
    X_train = X_train[X_train[:,0] >= 0]
        
    # Remove non-finite values
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    
    # dump it
    if outFile != None:
        jb.dump(X_train, outFile, compress=2)
    
    return X_train, fnames




def get_training_tiles(folder, label_field="training",
                     rgb=True, outFile=None, k=5, feattype='cgal',
                     add_fields=None, parallel=False, nt=-1):
    
    """ 
    Get training from multiple point clouds in a directory
    Features are calculated on the fly, so expect ~ 8mins per 6.4 million points
    (typically that's 53 features per point...)
    
    
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
    
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
        if std features a list of ints eg [20, 40, 60]
    
    feattype: string
               feature type either cgal or std

    add_fields: list of strings
        additional fields to be included as features
        e.g. for a LiDAR file ['Intensity', 'NumberOfReturns']
        
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
    
    if parallel == True:
        trainlist = Parallel(n_jobs=nt, verbose=2)(delayed(get_training_ply)(f,
                             label_field=label_field, 
                             rgb=rgb, outFile=None, k=k, 
                             add_fields=add_fields) for f in plylist)
        # using * in zip means it does the inverse of what it'd usually be used for
        trainlist, flist = zip(*trainlist)
        fnames = flist[0]
        del flist
    
    else:
    
        for f in plylist:
            
            train, fnames = get_training_ply(f, label_field=label_field, 
                         rgb=rgb, outFile=None, k=k, add_fields=add_fields)
        
            trainlist.append(train)
            del train
    
    X_train = np.vstack(trainlist)
    
    if outFile != None:
        jb.dump(X_train, outFile, compress=2)
    
    return X_train, fnames
    
    

def classify_ply(incld, inModel, class_field='label',
                 rgb=True, outcld=None, feattype='cgal', k=5,
                 add_fields=None):
    
    """ 
    Classify a point cloud (ply format) with amodel generated with this lib
    
    Features MUST match exactly of course, including add_fields
    
    As with previous funcs features are calculated on the fly to avoid huge files,
    so expect processing of ~6 million points (*53 features with k=5) to take
    8 minutes + classification time
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    inModel: string
          the input point cloud
    
                
    class_field: string
               the name of the field that the results will be written to
               this could already exist, if not will be created
               
    rgb: bool
        whether there is rgb data to be included
                 
    outcld: string
               path to a new ply to write if not writing to the input one
               
    feattype: string
               feature type previously used either cgal or std
               
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
        if std features a list of ints eg [20, 40, 60]

    add_fields: list of strings
        additional fields to be included as features
        e.g. for a LiDAR file ['Intensity', 'NumberOfReturns']

    """  
    
    
    # TODO - I/O classify by chunk?
    
    if feattype == 'cgal':
        featdf = cgal_features_mem(incld,  k=k, rgb=rgb, parallel=True)
    else:
        featdf = std_features(incld, outcld=None, k=k,
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None, tofile=False)

    
    pcd = PyntCloud.from_file(incld)
    
    if rgb == True:
        featdf['red'] = pcd.points['red'].to_numpy()
        featdf['green'] = pcd.points['green'].to_numpy()
        featdf['blue'] = pcd.points['blue'].to_numpy()
    
    if add_fields != None:
        # could be adding return no, intensity etc if it is LiDAR
        for a in add_fields:
            featdf[a] = pcd.points[a].to_numpy()
           
    X = featdf.to_numpy()
    
    X[np.where(np.isnan(X))]=0
    X = X[np.isfinite(X).all(axis=1)]
    del featdf
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
    
    pcd.points[class_field] = np.int32(predictClass)
    
    if outcld == None:    
        pcd.to_file(incld)
    else:
        pcd.to_file(outcld)


def classify_ply_tile(folder, inModel,  class_field='label',
                 rgb=True, k=5,  feattype='cgal', add_fields=None):
    
    """ 
    Classify a point clouds in a directory
    
    Features MUST match exactly of course, including add_fields
    
    As with previous funcs features are calculated on the fly to avoid huge files,
    so expect processing of ~6 million points (*53 features with k=5) to take
    8 minutes + classification time
    
     
    Parameters 
    ----------- 
    
    folder: string
              the input folder containing .ply files
                
    class_field: string
               the name of the field that the results will be written to
               this could already exist, if not will be created
    
    rgb: bool
        whether there is rgb data to be included
                 
    outcld: string
               path to a new ply to write if not writing to the input one
               
    rgb: bool
        whether there is rgb data to be included        
               
    feattype: string
               feature type previously used either cgal or std
               
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
        if std features a list of ints eg [20, 40, 60]
    
    add_fields: list of strings
        additional fields to be included as features
        e.g. for a LiDAR file ['Intensity', 'NumberOfReturns']
   

    """ 
            
            
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    for f in plylist:
        
        classify_ply(f, inModel, class_field=class_field,
                 rgb=rgb, outcld=None, feattype=feattype, k=k,
                 add_fields=add_fields)
        
        
def merge_train_points(folder, outcld):
    
    
    """
    Merge only the labeled points from various tiles to create a new point 
    cloud
    
    folder: string
              the input folder containing .ply files
                
    outcld: string
               the output ply file
    
    """
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    outlist = []
    
    for p in plylist:
        pcd = PyntCloud.from_file(p)
        df = pcd.points
        outlist.append(df[df.training>=0])
    
    # take the last pcd at random as the basis to write
    pcd.points = pd.concat(outlist)
    pcd.to_file(outcld)
    
def extract_train_points(incld, outcld):
    
    
    """
    Extract only the labeled points and save a new cloud
    
    incld: string
              the input ply
                
    outcld: string
               the output ply file
    
    """
    

    pcd = PyntCloud.from_file(incld)
    pcd.points = pcd.points[pcd.points.training>=0]
    
    # take the last pcd at random as the basis to write
    pcd.to_file(outcld)   

        
def create_model_cgal(incld, outModel, classes, k=5, rgb=False, method=None,
                      normal=False,
                      outcld=None,  ntrees=25, depth=20):
    
    """ 
    train a point cloud (ply format) with a model generated using CGAL
    (ETHZ Random forest})
                
    This will use the point cloud for evaluation and write the test labels
    
    The training must be labeled 'training' in the header 
                
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    outModel: string
          the input point cloud
          
    classes: list of strings
            the class labels in order
          
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
    
               
    rgb: bool
        whether there is rgb data to be included
    
    method: string
            default is None, otherwise smoothing ot graphcut
    
    normal: bool
            whether to include normals as features
                 
    outcld: string
               path to a new ply to write if not writing to the input one
    
    ntrees: int
               the no of trees for the ranndom forest
               
    depth: int
                depth of the random forest
               

    """ 
    points = Point_set_3(incld)
    print(points.size(), "points read")
    
    labels = Label_set()
    
    # make a list of swig object for later eval
    labelList = []
    for c in classes:
        labelList.append(labels.add(c))
    
    print('Calculating features...')
    features = Feature_set()
    generator = Point_set_feature_generator(points, k)
    # 5 is the number of levels (pyramids)
    
    # Not convince this is actually operating in parallel
    features.begin_parallel_additions()
    generator.generate_point_based_features(features)
    if normal == True and points.has_normal_map():
        generator.generate_normal_based_features(features, points.normal_map())
    
    if rgb is True:
        if points.has_int_map("red") and points.has_int_map("green") and points.has_int_map("blue"):
            generator.generate_color_based_features(features,
                                                    points.int_map("red"),
                                                    points.int_map("green"),
                                                    points.int_map("blue"))
    features.end_parallel_additions()
    
    classification = points.int_map("label")
    if not classification.is_valid():
        print("No ground truth found. Exiting.")
        exit()
    
    # Copy classification in training map for later evaluating
    training = points.add_int_map("training")
    for idx in points.indices():
        training.set(idx, classification.get(idx))
    
    print("Training random forest classifier...")
    classifier = ETHZ_Random_forest_classifier(labels, features)
    
    classifier.train(points.range(training), num_trees=ntrees, max_depth=depth)
    
    print("Saving model...")
    classifier.save_configuration(outModel)
    
    # classification map will be overwritten

    if method == 'graphcut':
        print("Classifying with graphcut...")
        classify_with_graphcut(points, labels, classifier,
                               generator.neighborhood().k_neighbor_query(6),
                               0.5,  # strength of graphcut
                               12,   # nb subdivisions (speed up)
                               classification)
    elif method == "smoothing":
        print("Classifying with local smoothing...")
        classify_with_local_smoothing(points, labels, classifier,
                                      generator.neighborhood().k_neighbor_query(6),
                                      classification)
    else:
        print("Classifying with standard algo")
        classify(points, labels, classifier, classification)

    if outcld == None:
        outcld = incld
    points.write(outcld)
    
    print("Evaluation:")
    evaluation = Evaluation(labels, points.range(training), points.range(classification))

    print(" * Accuracy =", evaluation.accuracy())
    print(" * Mean F1 score =", evaluation.mean_f1_score())
    print(" * Mean IoU =", evaluation.mean_intersection_over_union())
    
    print("Per label evaluation:")
    
    for label in labelList:
        print(" *", label.name(), ": precision =", evaluation.precision(label),
              " recall =", evaluation.recall(label),
              " iou =", evaluation.intersection_over_union(label))
    
    #TODO return the classif summary
    return 
        
        

def classify_cgal(incld, inModel, classes, k=5, rgb=False, method=None, 
                  normal=False, outcld=None):
    
    """ 
    classify a point cloud (ply format) with a model generated using CGAL
    (ETHZ Random forest})
                
    An evaluation will be printed
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    outModel: string
          the input point cloud
          
    classes: list of strings
            the class labels in order
          
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
    
               
    rgb: bool
        whether there is rgb data to be included
    
    method: string
            default is None, otherwise smoothing ot graphcut
    
    normal: bool
            whether to include normals as features
                 
    outcld: string
               path to a new ply to write if not writing to the input one
    
               

    """  
    
    
    points = Point_set_3(incld)
    print(points.size(), "points read")
    
    labels = Label_set()
    
    # make a list of swig object for later eval
    labelList = []
    for c in classes:
        labelList.append(labels.add(c))
    
    features = Feature_set()
    generator = Point_set_feature_generator(points, k)
    # 5 is the number of levels (pyramids)
    
    print('generating features...')
    features.begin_parallel_additions()
    generator.generate_point_based_features(features)
    if normal == True and points.has_normal_map():
        generator.generate_normal_based_features(features, points.normal_map())
    
    if rgb is True:
        if points.has_int_map("red") and points.has_int_map("green") and points.has_int_map("blue"):
            generator.generate_color_based_features(features,
                                                    points.int_map("red"),
                                                    points.int_map("green"),
                                                    points.int_map("blue"))
    features.end_parallel_additions()

    
    classification = points.add_int_map("label")
    
    classifier = ETHZ_Random_forest_classifier(labels, features)
    
    classifier.load_configuration(inModel)
    
    # classification map will be overwritten

    if method == 'graphcut':
        print("Classifying with graphcut...")
        classify_with_graphcut(points, labels, classifier,
                               generator.neighborhood().k_neighbor_query(6),
                               0.5,  # strength of graphcut
                               12,   # nb subdivisions (speed up)
                               classification)
    elif method == "smoothing":
        print("Classifying with local smoothing...")
        classify_with_local_smoothing(points, labels, classifier,
                                      generator.neighborhood().k_neighbor_query(6),
                                      classification)
    else:
        print("Classifying with standard algo")
        classify(points, labels, classifier, classification)
    
    
    print("Saving")
    
    if outcld == None:
        outcld = incld
    points.write(outcld)

def fix_cgal_classes(incld, clsdict, outcld=None):
    
    """
    Fix cgal classes wrongly relabelled* from the ply header
    
    *This can happen if you reopen a previously labelled file and add to it
    
    incld: string
           input ply
           
    clsdict: dict
            dict of correct correspondence {'tree': 1, 'building': 2}
            
    outcld: string
           optional output ply, otherwise save to input
    """
    
    
    pcd = PyntCloud.from_file(incld)
    #pcd.points.training.unique()
    #pcd.comments # gives header with training info
    
    # if using cgal, the class names are recorded in the header, by going from
    # index 2 we include unclassified, which is required
    potlabels = pcd.comments[2:]
    #we want the third word in each....assume this is always the case
    labels = [p.split()[2] for p in potlabels]
    # we also want a dict
    ints = [int(p.split()[1]) for p in potlabels]
    
    # make a dict to get a column of class names
    middict = dict(zip(ints, labels))
    
    pcd.points['trnname'] = pcd.points['training'].map(middict)
    # eg for the final conversion
    # clsnms = {'unclassified': -1 , 'FRJ': 0, 'BOF': 1, 'MAT': 2,
    #          'SHH': 3, 'ROC': 4, 'SHR': 5, 'AGU': 6}
    
    # and finally convert the training labels back
    pcd.points['training'] = pcd.points['trnname'].map(clsdict)
    
    #TODO  change the header comments??
    #pcd.points.drop(columns='trnname', inplace=True)
    
    if outcld == None:
        outcld = incld
    # BUG with Pyntcld screws up extent etc
    #pcd.to_file(outcld)
    
    # an array from which to write
    arr = pcd.points.training.to_numpy()
    
    del pcd
    
    points = Point_set_3(incld)
    
    clsmap = points.add_int_map('training')
    
    # TODO this is not ideal looping in python
    for i, p in enumerate(points.indices()):
        clsmap.set(p, int(arr[i]))
    
    points.write(outcld)
    
    

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





