from joblib import load, dump
import os
import notebook.config as config

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
        #Loading all base estimators from the  subfolders. These should go into
        #loaded_model.base_estimator, loaded_model.model.estimators_ ,
        #loaded_model.model.estimators and loaded_model.model.named_estimators_
        for k, clf in enumerate(loaded_model.base_estimators):
            #loading the base estimators models
            base_model = load_classifier(name + os.sep + clf[0])
            
            #Casting them into the necessary attributes
            #in base_estimators
            loaded_model.base_estimators[k] = (clf[0], base_model)
            #in model.estimators_
            if isinstance(loaded_model.model.estimators_[0], tuple):
                loaded_model.model.estimators_[k] = (loaded_model.model.estimators_[k][0], base_model)
            else:
                loaded_model.model.estimators_[k] = base_model
            #in model.estimators   
            if isinstance(loaded_model.model.estimators[0], tuple):
                loaded_model.model.estimators[k] = (loaded_model.model.estimators[k][0], base_model)
            #in model.named_estimators_    
            keyname = list(loaded_model.model.named_estimators_.keys())[k]
            loaded_model.model.named_estimators_[keyname] = base_model
            
    return loaded_model
    
    
    
    
            