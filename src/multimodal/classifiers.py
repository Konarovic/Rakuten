from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vit_keras import vit

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

import os

import src.config as config


def build_multi_model(txt_base_model, img_base_model, from_trained=None, max_length=256, img_size=(224, 224, 3),
                      num_class=27, drop_rate=0.0, activation='softmax'):
    """_summary_

    Args:
        base_name (str, optional): _description_. Defaults to 'camembert-base'.
    """
    #Bert branch    
    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype='int32', name='attention_mask')

    #Bert transformer model
    txt_base_model._name = 'bert_layers'
    txt_transformer_layer = txt_base_model({'input_ids': input_ids, 'attention_mask': attention_mask})
    txt_output = txt_transformer_layer[0][:, 0, :]
    # x = Dense(128, activation='relu', name='Dense_txt_1')(x)
    # x = Dropout(rate=drop_rate, name='Drop_out_top_1')(x)
    txt_model = Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask}, outputs=txt_output)
    
    #ViT transformer model
    input_img = Input(shape=img_size, name='inputs')
    img_output = img_base_model(input_img)
    img_model = Model(inputs=input_img, outputs=img_output)
    
    #Concatenate text and image models
    x = Concatenate()([txt_model.output, img_model.output])

    #Dense layers for classification
    x = Dropout(rate=drop_rate)(x)
    x = Dense(units=128, activation='relu', name='Dense_multi_1')(x)
    outputs = Dense(units=num_class, activation=activation, name='multi_classification_layer')(x)

    model = Model(inputs=[txt_model.input, img_model.input], outputs=outputs)
    
    if from_trained is not None:
        if isinstance(from_trained, dict):
            if 'text' in from_trained.keys():
                txt_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained['text'])
                print("loading weights for BERT from ", from_trained['text'])
                model.load_weights(txt_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
            if 'image' in from_trained.keys():
                img_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained['image'])
                print("loading weights for ViT from ", from_trained['image'])
                model.load_weights(img_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
        else:
            model_path = os.path.join(config.path_to_models, 'trained_models', from_trained)
            print("loading weights for multimodal model from ", from_trained)
            model.load_weights(model_path + '/weights.h5', by_name=True, skip_mismatch=True)
        
    
    return model

from keras.utils import Sequence

class MultimodalDataGenerator(Sequence):
    def __init__(self, img_data_generator, img_path, text_tokenized, labels, batch_size=32, target_size = (224, 224), shuffle=True):
        self.img_data_generator = img_data_generator
        self.dataframe = pd.DataFrame({'filename':img_path})#dataframe.copy()
        self.text_tokenized = text_tokenized
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:min((index + 1) * self.batch_size, len(self.dataframe))]
        batch_indexes_tensor = tf.convert_to_tensor(batch_indexes, dtype=tf.int32)
        
        batch_df = self.dataframe.iloc[batch_indexes]
        
        img_generator = self.img_data_generator.flow_from_dataframe(dataframe=batch_df, target_size=self.target_size,
                                                                    x_col="filename", y_col=None,class_mode=None,
                                                                    batch_size=len(batch_df), shuffle=False)
        
        images = np.concatenate([img_generator.next() for _ in range(len(img_generator))], axis=0)
        
        token_ids = tf.gather(self.text_tokenized['input_ids'], batch_indexes_tensor, axis=0)
        attention_mask = tf.gather(self.text_tokenized['attention_mask'], batch_indexes_tensor, axis=0)
        
        labels = self.labels[batch_indexes].values
        
        return [{"input_ids": token_ids, "attention_mask": attention_mask}, images], labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class TFmultiClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, txt_base_name='camembert-base', img_base_name='b16', from_trained = None, 
                 max_length=256, img_size=(224, 224, 3), augmentation_params=None,
                 num_class=27, drop_rate=0.2,
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
        # path to locally saved huggingface Bert model
        txt_base_model_path = os.path.join(config.path_to_models, 'base_models', txt_base_name)
        
        #Loading bert model base
        if not os.path.isdir(txt_base_model_path):
            # If the hugginface pretrained Bert model hasn't been yet saved locally, 
            # we load and save it from HuggingFace
            txt_base_model = TFAutoModel.from_pretrained(txt_base_name)
            txt_base_model.save_pretrained(txt_base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(txt_base_name)
            self.tokenizer.save_pretrained(txt_base_model_path)
        else:
            txt_base_model = TFAutoModel.from_pretrained(txt_base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(txt_base_name)
        
        #Loading ViT model base    
        default_action = lambda: print("img_base_name should be one of: b16, b32, L16 or L32")
        img_base_model = getattr(vit, 'vit_' + img_base_name, default_action)\
                                    (image_size = img_size[0:2], pretrained = True, 
                                     include_top = False, pretrained_top = False)
        
        self.model = build_multi_model(txt_base_model=txt_base_model, img_base_model=img_base_model,
                                       from_trained=from_trained, max_length=max_length, img_size=img_size,
                                       num_class=num_class, drop_rate=drop_rate, activation='softmax')
        
        self.max_length = max_length
        self.img_size = img_size
        self.txt_base_name = txt_base_name
        self.img_base_name = img_base_name
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
            dataset = self._getdataset(X, y, training=True)
            
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.history = self.model.fit(dataset, epochs=self.epochs, callbacks=self.callbacks)
        else:
            self.history = []
            
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        dataset = self._getdataset(X, training=False)
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)
    
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for each input.
        """
        dataset = self._getdataset(X, training=False)
        probs = self.model.predict(dataset)
        
        return probs
    
    
    def _getdataset(self, X, y=None, training=False):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            training (bool, optional): _description_. Defaults to False.
        """
        if y is None:
            y = 0
            
        df = pd.DataFrame({'labels': y, 'tokens': X['tokens'], 'img_path': X['img_path']})
        
        if training:
            shuffle = True
            params = self.augmentation_params
        else:
            shuffle = False
            params = dict(rotation_range=0, width_shift_range=0,
                          height_shift_range=0, horizontal_flip=False,
                          fill_mode='constant', cval=255)

        #Data generator for the train and test sets
        img_generator = ImageDataGenerator(rescale = 1./255, samplewise_center = True, samplewise_std_normalization = True,
                                            rotation_range=params['rotation_range'], 
                                            width_shift_range=params['width_shift_range'], 
                                            height_shift_range=params['height_shift_range'],
                                            horizontal_flip=params['horizontal_flip'],
                                            fill_mode=params['fill_mode'],
                                            cval=params['cval'])
        
        X_tokenized = self.tokenizer(df['tokens'].tolist(), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="tf")
        
        dataset = MultimodalDataGenerator(img_generator, df['img_path'], X_tokenized, df['labels'], 
                                          batch_size=self.batch_size, target_size = self.img_size[:2], shuffle=shuffle)
        
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

