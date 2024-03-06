from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

import os

import src.config as config


def build_bert_model(base_model, from_trained = None, max_length=256, num_class=27, drop_rate=0.0, activation='softmax'):
    """_summary_

    Args:
        base_name (str, optional): _description_. Defaults to 'camembert-base'.
    """
    #Input layer and tokenizer layer    
    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype='int32', name='attention_mask')

    #Base transformer model
    base_model._name = 'bert_layers'
    transformer_layer = base_model({'input_ids': input_ids, 'attention_mask': attention_mask})
    x = transformer_layer[0][:, 0, :]

    #Classification head
    x = Dense(128, activation='relu', name='Dense_top_1')(x)
    x = Dropout(rate=drop_rate, name='Drop_out_top_1')(x)
    output = Dense(num_class, activation=activation, name='classification_layer')(x)
    
    # Construct the final model
    model = Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask}, outputs=output)
    
    if from_trained is not None:
        trained_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained)
        print("loading weights from ", from_trained)
        model.load_weights(trained_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
    
    return model



class TFbertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_name='camembert-base', from_trained = None, 
                 max_length=256, num_class=27, drop_rate=0.2,
                 epochs=1, batch_size=32, learning_rate=5e-5, callbacks=None):
        """_summary_

        Args:
            base_name (str, optional): _description_. Defaults to 'camembert-base'.
            from_trained (_type_, optional): _description_. Defaults to None.
            max_length (int, optional): _description_. Defaults to 256.
            num_class (int, optional): _description_. Defaults to 27.
            drop_rate (int, optional): _description_. Defaults to 0.
            activation (str, optional): _description_. Defaults to 'softmax'.
        """
        # path to locally saved huggingface model
        base_model_path = os.path.join(config.path_to_models, 'base_models', base_name)
        
        # If the hugginface pretrained model hasn't been yet saved locally, 
        # we load and save it from HuggingFace
        if not os.path.isdir(base_model_path):
            print("loading from Huggingface")
            base_model = TFAutoModel.from_pretrained(base_name)
            base_model.save_pretrained(base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_name)
            self.tokenizer.save_pretrained(base_model_path)
        else:
            print("loading from Local")
            base_model = TFAutoModel.from_pretrained(base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        self.model = build_bert_model(base_model=base_model, from_trained = from_trained, max_length=max_length, num_class=num_class,
                                      drop_rate=drop_rate, activation='softmax')
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                   
        self.max_length = max_length
        self.base_name = base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
            dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            self.history = self.model.fit(dataset, epochs=self.epochs, callbacks=self.callbacks)
        else:
            self.history = []
            
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        dataset = self._preprocess(X, training=False)
        dataset = dataset.batch(self.batch_size)
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)
    
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for each input.
        """
        dataset = self._preprocess(X, training=False)
        dataset = dataset.batch(self.batch_size)
        probs = self.model.predict(dataset)
        
        return probs
    
    
    def _preprocess(self, X, y=None, training=False):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            training (bool, optional): _description_. Defaults to False.
        """
        X_tokenized = self.tokenizer(X['tokens'].tolist(), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
        
        if training:
            dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']}, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices({"input_ids": X_tokenized['input_ids'], "attention_mask": X_tokenized['attention_mask']})
        
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

