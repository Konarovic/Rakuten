from joblib import load, dump
import os
import src.config as config

import tensorflow as tf

from src.text.classifiers import TFbertClassifier, MLClassifier
from src.image.classifiers import ImgClassifier
from src.multimodal.classifiers import TFmultiClassifier, MetaClassifier

def load_classifier(name, parallel_gpu=False):
    """
    Loads a model from the directory specified in src.config file (config.path_to_models).

    Arguments:
    * name: The name of the saved model to load.
    * parallel_gpu: Flag to indicate whether to initialize the model 
        for parallel GPU usage.
    """
    #path to the directory where the model to load was saved
    model_path = os.path.join(config.path_to_models, 'trained_models', name)
    
    #Loading the model from there
    loaded_model = load(os.path.join(model_path, 'model.joblib'))
    
    if isinstance(loaded_model, (TFbertClassifier, ImgClassifier, TFmultiClassifier)):
        #tf.distribute.MirroredStrategy is not saved by joblib
        #so we need to update it here
        if parallel_gpu:
            loaded_model.strategy = tf.distribute.MirroredStrategy()
        else:
            loaded_model.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        #Re-building the model and loading the weights which has been saved
        # in model_path
        if isinstance(loaded_model, TFbertClassifier):
            loaded_model.model, loaded_model.tokenizer = loaded_model._getmodel(name)
        elif isinstance(loaded_model, ImgClassifier):
            loaded_model.model, loaded_model.preprocessing_function = loaded_model._getmodel(name)
        elif isinstance(loaded_model, TFmultiClassifier):
            loaded_model.model, loaded_model.tokenizer, loaded_model.preprocessing_function = loaded_model._getmodel(name)
            
    if isinstance(loaded_model, MetaClassifier):
        #Loading all base estimators from the  subfolders
        if isinstance(loaded_model.estimators_[0], tuple):
            for k, clf in enumerate(loaded_model.estimators_):
                base_model = load(os.path.join(model_path, clf[0], 'model.joblib'))
                base_model.load(name + os.sep + clf[0])
                loaded_model.estimators_[k] = (clf[0], base_model)
        else:
            for k in range(len(loaded_model.estimators_)):
                base_model = load(os.path.join(model_path, str(k), 'model.joblib'))
                base_model.load(name + os.sep + str(k))
                loaded_model.estimators_[k] = base_model
            
    return loaded_model
    
    