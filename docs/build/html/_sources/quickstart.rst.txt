.. _quickstart:

Quickstart
==========


An example workflow 
-------------------

The following attributes a point cloud with point-based features, trains a ML algorithm then classifys the pointcloud. 

.. code-block:: python

	from pointutils import learning as l
	from pointutils.props import std_features

	incld = "path/to/myfile.ply"

Attribute a pointcloud with features derived from PyntCloud at multiple scales. 

.. code-block:: python

	std_features(incld, outcld=None, k=[50,100,200]




We return a train variable and fnames (feature names for later reference). 

.. code-block:: python

	train, fnames = l.get_training_ply(outcloud, label_field='training', rgb=True, 
		                   outFile='cgaltrain.gz',
		                   ignore=ignr)


Standard grid search - we define a dict of parameters and a name for our output model. These parameters are for xgbboost. 

.. code-block:: python
 
	param_grid={'n_estimators': [200],
		                    'learning_rate': [0.1, 0.2, 0.3, 0.5], 
		                    'max_depth': [4, 6, 8, 10],
		                    'colsample_bytree': [0.4,0.6,0.8,1.0]}
	modelcg = 'cgaltestgb.gz'

	error_rate, best_param = l.create_model(train, modelcg, clf='xgb', params=param_grid, cv=5, cores=20, ply=True)

.. code-block:: python
	
    l.classify_ply(outcloud, modelcg, rgb=True, 
               outcld='classifGB100.ply', 
               ignore=ignr)

Finally plot the feature importance for some insight

.. code-block:: python

	l.plot_feature_importances(modelcg, fnames)





