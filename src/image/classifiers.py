import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vit_keras import vit

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

import os

import src.config as config


def build_ViT_model(base_name='b16', from_trained = None, img_size=(224, 224, 3), num_class=27, drop_rate=0.0, activation='softmax'):
    
    default_action = lambda: print("base_name should be one of: b16, b32, L16 or L32")

    #Loading Vision Transformer model from vit_keras
    vit_model = getattr(vit, 'vit_' + base_name, default_action)\
                        (image_size = img_size[0:2], pretrained = True, 
                         include_top = False, pretrained_top = False)

    model = Sequential(name = 'vision_transformer_' + base_name)
    model.add(Input(shape=img_size, name='inputs'))
    model.add(vit_model)
    model.add(Dense(128, activation = 'relu', name='Dense_top_1'))
    model.add(Dropout(rate=drop_rate, name='Drop_out_top_1'))
    model.add(Dense(num_class, activation = activation, name='classification_layer'))
    
    if from_trained is not None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        trained_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained)
        print("loading weights from ", from_trained)
        model.load_weights(trained_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
    
    return model



class ViTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_name='b16', from_trained = None, 
                 img_size=(224, 224, 3), num_class=27, drop_rate=0.2,
                 epochs=1, batch_size=32, learning_rate=5e-5,
                 augmentation_params=None, callbacks=None):
        """_summary_

        Args:
            base_name (str, optional): _description_. Defaults to 'b16'.
            from_trained (_type_, optional): _description_. Defaults to None.
            img_size (tuple, optional): _description_. Defaults to (224, 224, 3).
            num_class (int, optional): _description_. Defaults to 27.
            drop_rate (float, optional): _description_. Defaults to 0.2.
            epochs (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 32.
            learning_rate (_type_, optional): _description_. Defaults to 5e-5.
            callbacks (_type_, optional): _description_. Defaults to None.
        """
        
        self.model = build_ViT_model(base_name=base_name, from_trained = from_trained, img_size=img_size, num_class=num_class,
                                     drop_rate=drop_rate, activation='softmax')
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.img_size = img_size
        self.base_name = base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if augmentation_params is not None:
            self.augmentation_params = augmentation_params
        else:
            self.augmentation_params = dict(rotation_range=20, width_shift_range=0.1,
                                            height_shift_range=0.1, horizontal_flip=True,
                                            fill_mode='constant', cval=255)
        self.callbacks = callbacks
        
        if from_trained is not None:
            self.is_fitted_ = True
        
        
        
        
    def fit(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            epochs (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 32.
            learning_rate (_type_, optional): _description_. Defaults to 5e-5.

        Returns:
            _type_: _description_
        """
        
        if self.epochs > 0:
            dataset = self._preprocess(X, y, training=True)
            self.history = self.model.fit(dataset, epochs=self.epochs, callbacks=self.callbacks)
        else:
            self.history = []
            
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        dataset = self._preprocess(X, training=False)
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)
    
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for each input.
        """
        dataset = self._preprocess(X, training=False)
        probs = self.model.predict(dataset)
        return probs
    
    
    def _preprocess(self, X, y=None, training=False):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            training (bool, optional): _description_. Defaults to False.
        """
        if y is None:
            y = 0
            
        df = pd.DataFrame({'labels': y, 'img_path': X['img_path']})

        
        if training:
            shuffle = True
            params = self.augmentation_params
        else:
            shuffle = False
            params = dict(rotation_range=0, width_shift_range=0,
                          height_shift_range=0, horizontal_flip=False,
                          fill_mode='constant', cval=255)

        #Data generator for the train and test sets
        data_generator = ImageDataGenerator(rescale = 1./255, samplewise_center = True, samplewise_std_normalization = True,
                                            rotation_range=params['rotation_range'], 
                                            width_shift_range=params['width_shift_range'], 
                                            height_shift_range=params['height_shift_range'],
                                            horizontal_flip=params['horizontal_flip'],
                                            fill_mode=params['fill_mode'],
                                            cval=params['cval'])

        dataset = data_generator.flow_from_dataframe(dataframe=df, x_col='img_path', y_col='labels',
                                                     class_mode='raw', target_size=self.img_size[:2],
                                                     batch_size=self.batch_size, shuffle=shuffle)
        
        return dataset
    
    
    
    def save(self, name):
        """_summary_

        Args:
            dirpath (_type_): _description_
            name (_type_): _description_
        """
        #path to the directory where the model will be saved
        save_path = os.path.join(config.path_to_models, 'trained_models', name)
        
        #Creating it if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        #Saving model's weights to that location
        self.model.save_weights(os.path.join(save_path, 'weights.h5'))