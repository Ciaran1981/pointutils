# -*- coding: utf-8 -*-
"""
learning module 

Description
-----------
machine learning applied to point clouds

"""

# This must go first or it causes the error:
#ImportError: dlopen: cannot load any more object with static TLS
from xgboost.sklearn import XGBClassifier, XGBRegressor
import xgboost as xgb


from tqdm import tqdm

import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
from sklearn import svm
from osgeo import gdal, ogr#,osr
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, GroupKFold, KFold, 
                                     train_test_split,GroupShuffleSplit,
                                     StratifiedGroupKFold, 
                                     PredefinedSplit)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,RandomForestRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor,
                              VotingRegressor, VotingClassifier, StackingRegressor,
                              StackingClassifier,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, PowerTransformer,StandardScaler,
                                   QuantileTransformer)
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from catboost import (CatBoostClassifier, CatBoostRegressor,Pool )
import lightgbm as lgb
# from autosklearn.classification import AutoSklearnClassifier
# from autosklearn.regression import AutoSklearnRegressor
from skorch import NeuralNetClassifier
from optuna.integration import LightGBMPruningCallback
from optuna import create_study
import torch.nn as nn
import joblib

from pointutils.props import cgal_features_mem, std_features
gdal.UseExceptions()
ogr.UseExceptions()

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


def _group_cv(X_train, y_train, group, test_size=0.2, cv=10, strat=False):
    
    """
    Return the splits and and vars for a group grid search
    """
        
    # maintain group based splitting from initial train/test split
    # to main train set
    # TODO - sep func?
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=0)
    split = splitter.split(X_train, y_train, group)
    train_inds, test_inds = next(split)

    X_test = X_train[test_inds]
    y_test = y_train[test_inds]
    X_train = X_train[train_inds]
    y_train = y_train[train_inds]
    group_trn = group[train_inds]
    
    if strat == True:
        group_kfold = StratifiedGroupKFold(n_splits=cv).split(X_train,
                                                              y_train,
                                                              group_trn)
    else:
        group_kfold = GroupKFold(n_splits=cv).split(X_train,
                                                    y_train,
                                                    group_trn) 
    
    # all this not required produces same as above - keep for ref though
    # # Create a nested list of train and test indices for each fold
    # k_kfold = group_kfold.split(X_train, y_train, groups=group_trn)  

    # train_ind2, test_ind2 = [list(traintest) for traintest in zip(*k_kfold)]

    # cv = [*zip(train_ind2, test_ind2)]
    
    return X_train, y_train, X_test, y_test, group_kfold

def rec_feat_sel(X_train, featnames, preproc=('scaler', None),  clf='erf',  
                 group=None, 
                 cv=5, params=None, cores=-1, strat=True, 
                 test_size=0.3, regress=False, return_test=True,
                 scoring=None, class_names=None, save=True, cat_feat=None):
    
    """
    Recursive feature selection
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
              If the groupkfold is used, the last column will be the group labels
    
    
    params: dict
            a dict of model params (see scikit learn). If using a pipe(line)
            remember to prefix each param as follows with parent object 
            and two underscores.
            param_grid ={"selector__threshold": [0, 0.001, 0.01],
             "classifier__n_estimators": [1075]}
             
    clf: string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
    
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation          
    
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
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    
    bool index of features, list of chosen feature names
    """
    #TODO need to make all this a func
    # Not woring with hgb....
    clfdict = {'rf': RandomForestClassifier(random_state=0),
               'erf': ExtraTreesClassifier(random_state=0),
               'gb': GradientBoostingClassifier(random_state=0),
               'xgb': XGBClassifier(random_state=0),
               'logit': LogisticRegression(),
               # 'catb': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
                                          # task_type="GPU",
                                          # devices='0:1'),
               'lgbm': lgb.LGBMClassifier(random_state=0),
 #use_best_model=True-needs non empty eval set
                'hgb': HistGradientBoostingClassifier(early_stopping=True,
                                                      random_state=0)}
    
    regdict = {'rf': RandomForestRegressor(random_state=0),
               'erf': ExtraTreesRegressor(random_state=0),
               'gb': GradientBoostingRegressor(early_stopping=True,
                                               random_state=0),
               'xgb': XGBRegressor(random_state=0),
               # 'catb': CatBoostRegressor(logging_level='Silent', 
               #                           random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
                                          # devices='0:1'),
               'lgbm': lgb.LGBMRegressor(random_state=0),

               'hgb': HistGradientBoostingRegressor(early_stopping=True,
                                                    random_state=0)}
    
    if regress is True:
        model = regdict[clf]
        if clf == 'hgb':
            # until this is fixed - enabling above does nothing
            model.do_early_stopping = True
        if scoring is None:
            scoring = 'r2'
    else:
        model = clfdict[clf]
        cv = StratifiedKFold(cv)
        if scoring is None:
            scoring = 'accuracy'
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #no_classes = len(np.unique(y_train))
    

    # this is not a good way to do this
    # Does this matter for feature selection??
    if group is not None:
        
        X_train, y_train, X_test, y_test, cv = _group_cv(X_train, y_train,
                                                         group, test_size,
                                                         cv)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    

        
    rfecv = RFECV(estimator=model, 
                  step=1, 
                  cv=cv, 
                  scoring=scoring,
                  n_jobs=cores) # suspect this is no of folds
    
    pipeline  = Pipeline([preproc,
                          ('selector', rfecv)])


    # VERY slow - but I guess it is grid searching feats first
    #rfecv
    pipeline.fit(X_train, y_train)

    # First experiment is to add this as a fixed part of the process at the start
    # as it will slow it down otherwise

    # featind = pipeline[1].support_ # gains the feat indices(bool)
    # featnmarr = np.array(featnames)
    # featnames_sel = featnmarr[featind==True].tolist()

    # as X_train has changed we cant select from it within here
    
    return pipeline
    
    
def create_model(X_train, outModel, clf='erf', group=None, random=False,
                 cv=5, params=None, pipe='default', cores=-1, strat=True, 
                 test_size=0.3, regress=False, return_test=True,
                 scoring=None, class_names=None, save=True, cat_feat=None):
    
    """
    Brute force or random model creating using scikit learn.
    
    Parameters
    ---------------   
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
              If the groupkfold is used, the last column will be the group labels
    
    outModel: string
               the output model path which is a gz file, if using keras it is 
               h5
    
    params: dict
            a dict of model params (see scikit learn). If using a pipe(line)
            remember to prefix each param as follows with parent object 
            and two underscores.
            param_grid ={"selector__threshold": [0, 0.001, 0.01],
             "classifier__n_estimators": [1075]}
            
    pipe: str,dict,None
            if 'default' will include a preprocessing pipeline consisting of
            StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler(),
            otherwise specify in this form 
            pipe = {'scaler': [StandardScaler(), MinMaxScaler(),
                  Normalizer()]}
            or None will not preprocess the data
             
    clf: string
          an sklearn or xgb classifier/regressor 
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb,
    
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation          
          
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
    
    return_test: bool
              return X_test and y_test along with results (last two entries
              in list)
    
    scoring: string
              a suitable sklearn scoring type (see notes)
    
    class_names: list of strings
                class names in order of their numercial equivalents
    
    Returns
    -------
    A list of:
        
    [grid.best_estimator_, grid.cv_results_, grid.best_score_, 
            grid.best_params_, classification_report, X_test, y_test]
    
        
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
    

    clfdict = {'rf': RandomForestClassifier(random_state=0),
               'erf': ExtraTreesClassifier(random_state=0),
               'gb': GradientBoostingClassifier(random_state=0),
               'xgb': XGBClassifier(random_state=0),
               'logit': LogisticRegression(),
               # 'catb': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42),
               # 'catbgpu': CatBoostClassifier(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
               #                            devices='0:1'),
               'lgbm': lgb.LGBMClassifier(random_state=0),

                'hgb': HistGradientBoostingClassifier(random_state=0),
                'svm': SVC(),
                'nusvc': NuSVC(),
                'linsvc': LinearSVC()}
    
    regdict = {'rf': RandomForestRegressor(random_state=0),
               'erf': ExtraTreesRegressor(random_state=0),
               'gb': GradientBoostingRegressor(random_state=0),
               'xgb': XGBRegressor(random_state=0),
               # 'catb': CatBoostRegressor(logging_level='Silent', 
               #                           random_seed=42),
               # 'catbgpu': CatBoostRegressor(logging_level='Silent', # supress trees in terminal
               #                            random_seed=42,
               #                            task_type="GPU",
               #                            devices='0:1'),
               'lgbm': lgb.LGBMRegressor(random_state=0),
               'hgb': HistGradientBoostingRegressor(random_state=0),
                'svm': SVR(),
                'nusvc': NuSVR(),
                'linsvc': LinearSVR()}
    
    if regress is True:
        model = regdict[clf]
        if scoring is None:
            scoring = 'r2'
    else:
        model = clfdict[clf]
        if group is None:
            cv = StratifiedKFold(cv)
        if scoring is None:
            scoring = 'accuracy'
    
    if cat_feat:
        model.categorical_features = cat_feat
    
    
    bands = X_train.shape[1]-1
    
    #X_train = X_train.transpose()
    X_train = X_train[X_train[:,0] != 0]
    
     
    # Remove non-finite values
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
    #no_classes = len(np.unique(y_train))
    

    # this is not a good way to do this
    if regress == True:
        strat = False # failsafe
        
    if group is not None: # becoming a mess

        X_train, y_train, X_test, y_test, cv = _group_cv(X_train, y_train,
                                                             group, test_size,
                                                             cv, strat=strat)        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
        #cv = StratifiedKFold(cv)
        
    
    if pipe == 'default':
        # the dict must be in order of proc to work hence this
        # none is included to ensure
        # non prep data is considered
        #  should add selector?? seems to produce a load of nan errors (or warnings)
        sclr = {'scaler': [StandardScaler(), MinMaxScaler(),
                           Normalizer(), MaxAbsScaler(), 
                           QuantileTransformer(output_distribution='uniform'),
                           #PowerTransformer(),
                           None]} 
        sclr.update(params)
        
    else:
        sclr = pipe.copy() # to stop the var getting altered in script
        sclr.update(params)
    
    sk_pipe = Pipeline([("scaler", StandardScaler()),
                        #("selector", None), lord knows why this fails on var thresh
                        ("classifier", model)])
        
    
    if random is True:
        # recall the model is in the pipeline
        grid = RandomizedSearchCV(sk_pipe, param_distributions=sclr, 
                                  n_jobs=cores, n_iter=20,  verbose=2)
    else:
        grid = GridSearchCV(sk_pipe,  param_grid=sclr, 
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=1)


    grid.fit(X_train, y_train)
    
    joblib.dump(grid.best_estimator_, outModel) 
    
    testresult = grid.best_estimator_.predict(X_test)
    
    if regress == True:
        regrslt = regression_results(y_test, testresult)
        results = [grid]
        
    else:
        crDf = hp.plot_classif_report(y_test, testresult, target_names=class_names,
                                      save=outModel[:-3]+'._classif_report.png')
        
        confmat = metrics.confusion_matrix(testresult, y_test, labels=class_names)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confmat,
                                      display_labels=class_names)
        disp.plot()
    
        # confmat = hp.plt_confmat(X_test, y_test, grid.best_estimator_, 
        #                          class_names=class_names, 
        #                          cmap=plt.cm.Blues, 
        #                          fmt="%d", 
        #                          save=outModel[:-3]+'_confmat.png')
        
        results = [grid, crDf, confmat]
        
    if return_test == True:
        results.extend([X_test, y_test])
    return results

def combine_models(X_train, modelist, mtype='regress', method='voting', group=None, 
                   test_size=0.3, outmodel=None, class_names=None, params=None,
                   final_est='xgb', cv=5):#, cores=1):
    
    """
    Combine models using either the voting or stacking methods in scikit-learn
    
    Parameters
    ----------
    
    X_train: np array
              numpy array of training data where the 1st column is labels.
    
    outmodel: string
               the output model path which is a gz file, if using keras it is 
               h5
    
    modelist: dict
            a list of tuples of model type (str) and model (class) e.g.
            [('gb', gmdl), ('rf', rfmdl), ('lr', reg3)]
    
    mtype: string
            either clf or regress
    
    method: string
            either voting or k = clusterer.labels_stacking

    test_size: float
            percentage to hold out to test  
                
    group: np.array
            array of group labels for train/test split and grid search
            useful to avoid autocorrelation
              
    class_names: list of strings
                class names in order of their numercial equivalents
    
    params: dict
            a dict of model params (If using stacking method for final estimator)
    
    final_est: string
             the final estimator one of (rf, gb, erf, xgb, logit)

    """
    bands = X_train.shape[1]-1
    
    X_train = X_train[X_train[:,0] != 0]
    
    if group is not None: #replace this later not good
        inds = np.where(np.isfinite(X_train).all(axis=1))
        group = group[inds]
    
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # y labels
    y_train = X_train[:,0]

    # remove labels from X_train
    X_train = X_train[:,1:bands+1]
    
        # Remove non-finite values
 
    
    # this is not a good way to do this
    if group is not None:
        # this in theory should not make any difference at this stage...
        # ...included for now
        # maintain group based splitting from initial train/test split
        # to main train set
        # TODO - sep func?
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=0)
        split = splitter.split(X_train, y_train, group)
        train_inds, test_inds = next(split)
    
        X_test = X_train[test_inds]
        y_test = y_train[test_inds]
        X_train = X_train[train_inds]
        y_train = y_train[train_inds]
        group_trn = group[train_inds]
        
        group_kfold = GroupKFold(n_splits=cv) 
        # Create a nested list of train and test indices for each fold
        k_kfold = group_kfold.split(X_train, y_train, group_trn)  

        train_ind2, test_ind2 = [list(traintest) for traintest in zip(*k_kfold)]

        cv = [*zip(train_ind2, test_ind2)]
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=0)
    
    if method == 'voting':
        comb = VotingRegressor(estimators=modelist)#, n_jobs=cores)
        # we only wish to predict really - but  necessary 
        # for sklearn model construct
    else:
        clfdict = {'rf': RandomForestClassifier(random_state=0),
                   'erf': ExtraTreesClassifier(random_state=0),
                   'gb': GradientBoostingClassifier(random_state=0),
                   'xgb': XGBClassifier(random_state=0),
                   'logit': LogisticRegression(),
                   'lgbm': lgb.LGBMClassifier(random_state=0),
                    'hgb': HistGradientBoostingClassifier(random_state=0),
                    'svm': SVC(),
                    'nusvc': NuSVC(),
                    'linsvc': LinearSVC()}
        
        regdict = {'rf': RandomForestRegressor(random_state=0),
                   'erf': ExtraTreesRegressor(random_state=0),
                   'gb': GradientBoostingRegressor(random_state=0),
                   'xgb': XGBRegressor(random_state=0),
                   'lgbm': lgb.LGBMRegressor(random_state=0),
                   'hgb': HistGradientBoostingRegressor(random_state=0),
                    'svm': SVR(),
                    'nusvc': NuSVR(),
                    'linsvc': LinearSVR()}
        
        if mtype == 'regress':
            # won't accept the dict even with the ** to unpack it
            fe = regdict[final_est]()
            fe.set_params(**params)
            
            comb = StackingRegressor(estimators=modelist,
                                final_estimator=fe)
        else:
            fe = clfdict[final_est]()
            fe.set_params(**params)
            cv = StratifiedKFold(cv)            
            comb = StackingClassifier(estimators=modelist,
                                final_estimator=fe)
            
    comb.fit(X_train, y_train)
    
    # Since there is no train/test this is misleading...
    train_pred = comb.predict(X_train)
    test_pred = comb.predict(X_test)  
    if mtype == 'regress':
        print('On the train split (not actually trained on this)')
        regression_results(y_train, train_pred)
        
        print('On the test split')
        regression_results(y_test, test_pred)
        
    else:
        crDf = hp.plot_classif_report(y_test, test_pred, target_names=class_names,
                              save=outmodel[:-3]+'._classif_report.png')
    
        confmat = hp.plt_confmat(X_test, y_test, comb, 
                                 class_names=class_names, 
                                 cmap=plt.cm.Blues, 
                                 fmt="%d", 
                                 save=outmodel[:-3]+'_confmat.png')
    
    if outmodel is not None:
        joblib.dump(comb, outmodel)
    return comb, X_test, y_test 


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('MedianAE', round(median_absolute_error, 4))
    print('RMSE: ', round(np.sqrt(mse),4))   
    #TODO add when sklearn updated    
    display = metrics.PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        #ax=ax,
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )

 

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
   ExtraTreesRegressor for label, clf_err in error_rate.items():
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
    Can use las also!
    
    
    Parameters 
    ----------- 
    
    incld: string
              the input point cloud
    
    label_field: string
              the name of the field representing the training points which must
              be positive integers
    
    feattype: string
                cgal (ply only), std(las or ply) or pdal (las only)
                (pdal feats must be written in advance at is assumed to be 'all')
              
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
    pcd = PyntCloud.from_file(incld)
    
    if feattype == 'cgal':
        featdf = cgal_features_mem(incld,  k=k, rgb=rgb, parallel=True)
    elif feattype == 'std':
        featdf = std_features(incld, outcld=None, k=k,
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None, tofile=False)
    elif feattype == 'pdal':
        props = ['linearity', 'planarity', 'scattering', 'verticality', 
                 'omnivariance', 'anisotropy','eigenentropy', 'eigenvaluesum',
                 'surfacevariation','demantkeverticality', 'density']
        
        featdf = pcd.points[props]
    
    
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
    Features are calculated on the fly (unless using pdal), 
    so expect ~ 8mins per 6.4 million points
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
                cgal (ply only), std(las or ply) or pdal (las only)
                (pdal feats must be written in advance at is assumed to be 'all')

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
    Classify a point cloud (ply format) with a model generated with this lib
    
    Features MUST match exactly of course, including add_fields
    
    As with previous funcs features are calculated on the fly to avoid huge files,
    so expect processing of ~6 million points (*53 features with k=5) to take
    8 minutes + classification time unless using pdal where they must be written
    to file in advance
    
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
               feature type previously used either cgal, std or pdal 
               (these have much in common just different ways of calculating)
               pdal features must have been written to file prior to this
               step
               
    k: int or list of ints
        the number of scales at which to calcualte features
        if cgal features an int eg 5
        if std features a list of ints eg [20, 40, 60]

    add_fields: list of strings
        additional fields to be included as features
        e.g. for a LiDAR file ['Intensity', 'NumberOfReturns']

    """  
    
    
    # TODO - I/O classify by chunk?
    pcd = PyntCloud.from_file(incld)
    
    if feattype == 'cgal':
        featdf = cgal_features_mem(incld,  k=k, rgb=rgb, parallel=True)
    elif feattype == 'std':
        featdf = std_features(incld, outcld=None, k=k,
                 props=['anisotropy', "curvature", "eigenentropy", "eigen_sum",
                         "linearity","omnivariance", "planarity", "sphericity"],
                        nrm_props=None, tofile=False)
    elif feattype == 'pdal':
        props = ['linearity', 'planarity', 'scattering', 'verticality', 'omnivariance', 'anisotropy',
       'eigenentropy', 'eigenvaluesum', 'surfacevariation',
       'demantkeverticality', 'density']
        
        featdf = pcd.points[props]
    
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

#TODO replace with an ept/entwine solution?
        
def classify_ply_tile(folder, inModel,  class_field='label',
                 rgb=True, k=5,  feattype='cgal', add_fields=None):
    
    """ 
    Classify a point clouds in a directory
    
    Features MUST match exactly of course, including add_fields
    
    As with previous funcs features are calculated on the fly to avoid huge files,
    so expect processing of ~6 million points (*53 features with k=5) to take
    8 minutes + classification time unless using pdal where they must be written
    to file in advance
    
     
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
               feature type previously used either cgal, std or pdal 
               (these have much in common just different ways of calculating)
               pdal features must have been written to file prior to this
               step
               
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
                      outcld=None,  ntrees=25, depth=20, cut_strngth=0.2,
                      n_sub=12):
    
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
          the output model
          
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
                               cut_strngth,  # strength of graphcut
                               n_sub,   # nb subdivisions (speed up)
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
    # return 
        
        

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
    
    inModel: string
          the input model
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





