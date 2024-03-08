""" 
Class implementations for building and utilizing multimodal and ensemble models, adeptly combining text and 
image data for comprehensive classification tasks. This documentation provides a summary of the 
TFmultiClassifier and MetaClassifier classes with their main paremeters and methods

* TFmultiClassifier Class
    A TensorFlow-based classifier for multimodal (text and image) data.

    Constructor Parameters:
        * txt_base_name: Identifier for the base text model.
        * img_base_name: Identifier for the base image model.
        * from_trained: Optional, specifies pre-trained model weights.
        * max_length: Maximum sequence length for text inputs.
        * img_size: The size of the input images.
        * augmentation_params: Data augmentation parameters.
        * validation_split: Fraction of data to be used for validation.
        * validation_data: Data to use for validation during training.
        * num_class: The number of output classes.
        * drop_rate: Dropout rate for regularization.
        * epochs: Number of epochs to train the model.
        * batch_size: Batch size for training.
        * learning_rate: Learning rate for the optimizer.
        * validation_split: Fraction of data used for validation.
        * callbacks: Callbacks for training.
        * parallel_gpu: Whether to use parallel GPU support.

    Methods:
        * fit(X, y): Trains the model.
        * predict(X): Predicts class labels for input data.
        * predict_proba(X): Predicts class probabilities.
        * classification_score(X, y): Calculates classification metrics.
        * save(name): Saves the model.
        * load(name, parallel_gpu): Loads a saved model.

    Example Usage:
        X = pd.DataFrame({'text': txt_data, 'img_path': img_data})
        y = labels
        classifier = TFmultiClassifier(txt_base_name='bert-base-uncased', img_base_name='vit_b16', epochs=5, batch_size=2)
        classifier.fit(X, y)
        f1score = classifier.classification_score(X_test, y_test)
        classifier.save('multimodal_model')
        
* MetaClassifier Class
    A wrapper class for various ensemble methods, enabling the combination of multiple classifier models for improved prediction accuracy.

    Constructor Parameters:
        * base_estimators: List of tuples with base estimators and their names.
        * method: The ensemble method to use, such as 'voting', 'stacking', 'bagging', or 'boosting'.
        * from_trained: Optional; path to a previously saved ensemble model.
        * **kwargs: Additional arguments specific to the chosen ensemble method.
    
    Methods:
        * fit(X, y): Trains the ensemble model on the given dataset.
        * predict(X): Predicts class labels for the input data.
        * predict_proba(X): Predicts class probabilities for the input data (if supported by the base models).
        * classification_score(X, y): Calculates the weighted F1-score for the predictions.
        * save(name): Saves the ensemble model and its base models.
        * load(name): Loads the ensemble model and its base models.
        
    Example Usage:
        base_estimators = [('clf1', TFbertClassifier()), ('clf2', MLClassifier())]
        meta_classifier = MetaClassifier(base_estimators=base_estimators, method='voting')
        meta_classifier.fit(X, y)
        predictions = meta_classifier.predict(X_test)
        f1score = classifier.classification_score(X_test, y_test)
        meta_classifier.save('my_ensemble_model')
        meta_classifier.load('my_ensemble_model')
"""


from transformers import TFAutoModel, AutoTokenizer, CamembertTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vit_keras import vit

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from joblib import load, dump
import os

import src.config as config


def build_multi_model(txt_base_model, img_base_model, from_trained=None, max_length=256, img_size=(224, 224, 3),
                      num_class=27, drop_rate=0.0, activation='softmax', attention_numheads=0,
                      attention_query='image', strategy=None):
    """
    Creates a multimodal classification model that combines text and image data for prediction tasks.

    Arguments:
    * txt_base_model: Pre-initialized text model (such as BERT) to be used for text feature extraction.
    * img_base_model: Pre-initialized image model (such as Vision Transformer, ViT) for image feature extraction.
    * from_trained (optional): Path or dictionary specifying the name of pre-trained models for text and/or image models. 
      If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
      full model should be passed as a simple string
    * max_length (int, optional): Maximum sequence length for text inputs. Default is 256.
    * img_size (tuple, optional): Size of the input images. Default is (224, 224, 3).
    * num_class (int, optional): Number of classes for the classification task. Default is 27.
    * drop_rate (float, optional): Dropout rate applied in the final layers of the model. Default is 0.0.
    * activation (str, optional): Activation function for the output layer. Default is 'softmax'.
    * attention_numheads (int, optional): number of cross-attention head before the classification layer
    * strategy: TensorFlow distribution strategy to be used during model construction.
    
    Returns:
    A TensorFlow Model instance representing the constructed multimodal classification model.
    
    Example usage:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    txt_model = TFAutoModel.from_pretrained('bert-base-uncased')
    img_model = vit.vit_b16(image_size=(224, 224), pretrained=True, include_top=False, pretrained_top=False)
    multimodal_model = build_multi_model(txt_base_model=txt_model, img_base_model=img_model, max_length=256, img_size=(224, 224, 3), num_class=10, strategy=strategy)
    """
    with strategy.scope():
        #Bert branch    
        input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(max_length,), dtype='int32', name='attention_mask')

        #Bert transformer model
        txt_base_model._name = 'txt_base_layers'
        txt_transformer_layer = txt_base_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        x = txt_transformer_layer[0][:, 0, :]
        x = LayerNormalization(epsilon=1e-6, name='txt_normalization')(x)
        
        x = Dense(128, activation = 'relu', name='txt_Dense_top_1')(x)
        x = Dropout(rate=drop_rate, name='txt_Drop_out_top_1')(x)
        outputs = Dense(num_class, activation= 'relu', name='txt_classification_layer')(x)
        # outputs = Dense(units=2*num_class, activation=activation, name='text_classification_layer')(x)
        txt_model = Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask}, outputs=outputs)
        
        #Loading pre-saved weights to the Bert model if provided
        if from_trained is not None:
            if isinstance(from_trained, dict):
                if 'text' in from_trained.keys():
                    txt_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained['text'])
                    print("loading weights for BERT from ", from_trained['text'])
                    txt_model.load_weights(txt_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
                
        
        #ViT transformer model
        input_img = Input(shape=img_size, name='inputs')
        img_base_model._name = 'img_base_layers'
        x = img_base_model(input_img)
        x = LayerNormalization(epsilon=1e-6, name='img_normalization')(x)
        
        x = Dense(128, activation = 'relu', name='img_Dense_top_1')(x)
        x = Dropout(rate=drop_rate, name='img_Drop_out_top_1')(x)
        outputs = Dense(num_class, activation='relu', name='img_classification_layer')(x)
        # outputs = Dense(units=2*num_class, activation=activation, name='img_classification_layer')(x)
        img_model = Model(inputs=input_img, outputs=outputs)
        
        #Loading pre-saved weights to the Image model if provided
        if from_trained is not None:
            if isinstance(from_trained, dict):
                if 'image' in from_trained.keys():
                    img_model_path = os.path.join(config.path_to_models, 'trained_models', from_trained['image'])
                    print("loading weights for ViT from ", from_trained['image'])
                    img_model.load_weights(img_model_path + '/weights.h5', by_name=True, skip_mismatch=True)
        
        #Concatenate text and image models
        if attention_numheads == 0:
            x = Concatenate()([txt_model.output, img_model.output])
        else:
            if attention_query == 'image':
                embed_dim = txt_model.output.shape[-1]
                attention_layer = MultiHeadAttention(num_heads=attention_numheads, key_dim=embed_dim, name='multi_multihead_layer')
                x = attention_layer(query=img_model.output, key=txt_model.output, value=txt_model.output)
            else:
                embed_dim = img_model.output.shape[-1]
                attention_layer = MultiHeadAttention(num_heads=attention_numheads, key_dim=embed_dim, name='multi_multihead_layer')
                x = attention_layer(query=txt_model.output, key=img_model.output, value=img_model.output)
            
        #Dense layers for classification
        x = Dropout(rate=drop_rate, name='multi_Drop_out_top_1')(x)
        # x = Dense(units=128, activation='relu', name='Dense_multi_1')(x)
        outputs = Dense(units=num_class, activation=activation, name='multi_classification_layer')(x)

        model = Model(inputs=[txt_model.input, img_model.input], outputs=outputs)
        
        #Loading pre-saved weights to the full model if provided
        if from_trained is not None:
            if not isinstance(from_trained, dict):
                model_path = os.path.join(config.path_to_models, 'trained_models', from_trained)
                print("loading weights for multimodal model from ", from_trained)
                model.load_weights(model_path + '/weights.h5', by_name=True, skip_mismatch=True)
        
    
    return model

from keras.utils import Sequence

class MultimodalDataGenerator(Sequence):
    """
    A custom data generator for batching and preprocessing multimodal data (text and images) for training or prediction with a Keras model.

    Constructor Arguments:
    * img_data_generator: An instance of Data genrator for real-time data augmentation and preprocessing of image and text data.
    * img_path: List or Pandas Series containing paths to the images.
    * text_tokenized: A dictionary containing tokenized text data with keys 'input_ids' and 'attention_mask'.
    * labels: Numpy array or Pandas Series containing target labels for the dataset.
    * batch_size (int, optional): Number of samples per batch. Default is 32.
    * target_size (tuple, optional): The dimensions to which all images found will be resized. Default is (224, 224).
    * shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Default is True.
    
    Methods:
    * __len__: Returns the number of batches per epoch.
    * __getitem__: Returns a batch of data (text and images) and corresponding labels.
    * on_epoch_end: Updates indexes after each epoch if shuffle is True.
    """
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
    """
    A TensorFlow-based classifier for multimodal (text and image) data, implementing the scikit-learn estimator interface.

    Constructor Arguments:
    * txt_base_name: Identifier for the base text model (e.g., 'bert-base-uncased').
    * img_base_name: Identifier for the base image model (e.g., 'vit_b16').
    * from_trained: Path or dictionary specifying the name of pre-trained models for text and/or image models. 
      If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
      full model should be passed as a simple string
    * max_length (int, optional): Maximum sequence length for text inputs.
    * img_size (tuple, optional): Size of the input images.
    * augmentation_params: Parameters for data augmentation.
    * validation_split: fraction of the data to use for validation during training. Default is 0.0.
    * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
    * num_class (int, optional): Number of classes for the classification task.
    * drop_rate (float, optional): Dropout rate applied in the final layers of the model.
    * epochs (int, optional): Number of epochs for training.
    * batch_size (int, optional): Number of samples per batch.
    * learning_rate (float, optional): Learning rate for the optimizer.
    * callbacks: List of Keras callbacks to be used during training.
    * parallel_gpu (bool, optional): Whether to use TensorFlow's parallel GPU training capabilities.
    
    Methods:
    * fit: Trains the multimodal model on a dataset.
    * predict: Predicts class labels for the given input data.
    * predict_proba: Predicts class probabilities for the given input data.
    * classification_score: Computes classification metrics for the given input data and true labels.
    * save: Saves the model's weights and tokenizer to the specified directory.
    * load: Loads the model's weights and tokenizer from the specified directory.
    
    Example Usage:
    X = pd.DataFrame({'text': txt_data, 'img_path': img_data})
    y = labels
    classifier = TFmultiClassifier(txt_base_name='bert-base-uncased', img_base_name='vit_b16', epochs=5, batch_size=2)
    classifier.fit(X, y)
    score = classifier.classification_score(X_test, y_test)
    classifier.save('multimodal_model')
    """
    
    def __init__(self, txt_base_name='camembert-base', img_base_name='vit_b16', from_trained = None, 
                 max_length=256, img_size=(224, 224, 3), augmentation_params=None,
                 num_class=27, drop_rate=0.2, epochs=1, batch_size=32, 
                 attention_numheads=0, attention_query='image',
                 validation_split=0.0, validation_data=None,
                 learning_rate=5e-5, callbacks=None, parallel_gpu=True):
        """
        Constructor: __init__(self, txt_base_name='camembert-base', img_base_name='b16', from_trained = None, 
                              max_length=256, img_size=(224, 224, 3), augmentation_params=None, num_class=27, drop_rate=0.2,
                              epochs=1, batch_size=32, learning_rate=5e-5, callbacks=None, parallel_gpu=True)
                              
        Initializes a new instance of the TFmultiClassifier.

        Arguments:

        * txt_base_name: The identifier for the base BERT model. Tested base model are 'camembert-base',
          'camembert/camembert-base-ccnet'. Default is 'camembert-base'.
        * img_base_name: The identifier for the base vision model architecture (e.g., 'vit_b16', 'vgg16', 'resnet50').
        * from_trained: Path or dictionary specifying the name of pre-trained models for text and/or image models. 
          If a dictionary, keys should be 'text' and 'image' with names of the models as values. Name of a pret-trained 
          full model should be passed as a simple string
        * max_length: The sequence length that the tokenizer will generate. Default is 256.
        * img_size: The size of input images.
        * num_class: The number of classes for the classification task. Default is 27.
        * augmentation_params: a dictionnary with parameters for data augmentation (see ImageDataGenerator).
        * drop_rate: Dropout rate for the classification head. Default is 0.2.
        * epochs: The number of epochs to train the model. Default is 1.
        * batch_size: Batch size for training. Default is 32.
        * learning_rate: Learning rate for the optimizer. Default is 5e-5.
        * validation_split: fraction of the data to use for validation during training. Default is 0.0.
        * validation_data: a tuple with (features, labels) data to use for validation during training. Default is None.
        * callbacks: A list of tuples with the name of a Keras callback and a dictionnary with matching
          parameters. Example: ('EarlyStopping', {'monitor':'loss', 'min_delta': 0.001, 'patience':2}).
          Default is None.
        * parallel_gpu: Flag to indicate whether to use parallel GPU support. Default is False.
        
        Returns:

        An instance of TFmultiClassifier.
        """
        
        #defining the parallelization strategy
        if parallel_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        #Defining attributes
        self.max_length = max_length
        self.img_size = img_size
        self.txt_base_name = txt_base_name
        self.img_base_name = img_base_name
        self.from_trained = from_trained
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.attention_numheads = attention_numheads
        self.attention_query = attention_query
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if augmentation_params is not None:
            self.augmentation_params = augmentation_params
        else:
            self.augmentation_params = dict(rotation_range=20, width_shift_range=0.1,
                                            height_shift_range=0.1, horizontal_flip=True,
                                            fill_mode='constant', cval=255)    
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.callbacks = callbacks
        self.parallel_gpu = parallel_gpu
        
        #Building model and tokenizer
        self.model, self.tokenizer, self.preprocessing_function = self._getmodel(from_trained)
        
        #For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted
        if from_trained is not None:
            self.is_fitted_ = True
            
            
    def _getmodel(self, from_trained=None):
        """
        Internal method to initialize or load the base model and set up preprocessing.
        """
        # path to locally saved huggingface Bert model
        txt_base_model_path = os.path.join(config.path_to_models, 'base_models', self.txt_base_name)
        
        with self.strategy.scope():
            #Loading bert model base
            if not os.path.isdir(txt_base_model_path):
                # If the hugginface pretrained Bert model hasn't been yet saved locally, 
                # we load and save it from HuggingFace
                txt_base_model = TFAutoModel.from_pretrained(self.txt_base_name)
                txt_base_model.save_pretrained(txt_base_model_path)
                tokenizer = CamembertTokenizer.from_pretrained(self.txt_base_name)
                tokenizer.save_pretrained(txt_base_model_path)
            else:
                txt_base_model = TFAutoModel.from_pretrained(txt_base_model_path)
                tokenizer = CamembertTokenizer.from_pretrained(txt_base_model_path)
            
            #Loading ViT model base    
            default_action = lambda: print("img_base_name should be one of: b16, b32, L16 or L32")
            img_base_model = getattr(vit, 'vit_' + self.img_base_name[-3:], default_action)\
                                        (image_size = self.img_size[0:2], pretrained = True, 
                                        include_top = False, pretrained_top = False)
            preprocessing_function = None
        
        model = build_multi_model(txt_base_model=txt_base_model, img_base_model=img_base_model,
                                       from_trained=from_trained, max_length=self.max_length, img_size=self.img_size,
                                       num_class=self.num_class, drop_rate=self.drop_rate, activation='softmax',
                                       attention_numheads=self.attention_numheads, attention_query=self.attention_query,
                                       strategy = self.strategy)
        
        return model, tokenizer, preprocessing_function
        
        
    def fit(self, X, y):
        """
        Trains the model on the provided dataset.
        
        Parameters:
        * X: The image and text data for training. Should be a dataframe with columns
          "tokens" containing the text and column "img_path" containing the full paths 
          to the images
        * y: The target labels for training.
        
        Returns:
        The instance of TFmultiClassifier after training.
        """
        
        if self.epochs > 0:
            # Initialize validation data placeholder
            dataset_val = None
            
            if self.validation_split > 0:
                #Splitting data for validation as necessary
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=123)
                #Fetching the dataset generator for validation
                dataset_val = self._getdataset(X_val, y_val, training=False)
            elif self.validation_data is not None:
                #If validation data are provided in self.validation_data, we fetch those
                dataset_val = self._getdataset(self.validation_data[0], self.validation_data[1], training=True)
                X_train, y_train = X, y
            else:
                # Use all data for training if validation split is 0
                X_train, y_train = X, y
                
            #Fetching the training dataset generator
            dataset = self._getdataset(X_train, y_train, training=True)
            
            with self.strategy.scope():
                #defining the optimizer
                optimizer = Adam(learning_rate=self.learning_rate)
                
                #Creating callbacks based on self.callback
                callbacks = []
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback_api = getattr(tf.keras.callbacks, callback[0])
                        callbacks.append(callback_api(**callback[1]))
                        
                #Compiling
                self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                #Fitting the model
                fit_args = {'epochs': self.epochs, 'callbacks': callbacks}
                if dataset_val is not None:
                    fit_args['validation_data'] = dataset_val
                    
                self.history = self.model.fit(dataset, **fit_args)
        else:
            #if self.epochs = 0, we just pass the model, considering it has already been trained
            self.history = []
        
        #For sklearn, adding attribute finishing with _ to indicate
        # that the model has already been fitted    
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Arguments:
        * X: The image and text data for prediction. Should be a dataframe with columns
          "tokens" containing the text and column "img_path" containing the full paths 
          to the images
        
        Returns:
        An array of predicted class labels.
        """
        dataset = self._getdataset(X, training=False)
        preds = self.model.predict(dataset)
        return np.argmax(preds, axis=1)
    
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.

        Arguments:
        * X: The image and text data for prediction. Should be a dataframe with columns
          "tokens" containing the text and column "img_path" containing the full paths 
          to the images
          
        Returns:
        An array of class probabilities for each input instance.
        """
        dataset = self._getdataset(X, training=False)
        probs = self.model.predict(dataset)
        
        return probs
    
    
    def _getdataset(self, X, y=None, training=False):
        """
        Internal method to prepare a TensorFlow dataset from the input data.
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
    
    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y
        
        Arguments:
        * X: The image and text data for which to predict labels. Should be 
          a dataframe with columns "tokens" containing the text and column
          "img_path" containing the full paths to the images
        * y: The target labels to predict.
        
        Returns:
        The average weighted f1-score. Also save scores in classification_results
        and f1score attributes
        """
        
        #predict class labels for the input text X
        pred = self.predict(X)
        
        #Save classification report
        self.classification_results = classification_report(y, pred)
        
        #Save weighted f1-score
        self.f1score = f1_score(y, pred, average='weighted')
        
        return self.f1score    
    
    
    def save(self, name):
        """
        Saves the model to the directory specified in src.config file (config.path_to_models).

        Arguments:
        * name: The name to be used for saving the model.
        """
        #path to the directory where the model will be saved
        save_path = os.path.join(config.path_to_models, 'trained_models', name)
        
        #Creating it if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        #Saving model's weights to that location
        self.model.save_weights(os.path.join(save_path, 'weights.h5'))
        
        #Saving the model except for keras objects which are not serialized 
        #by joblib
        model_backup = self.model
        tokenizer_backup = self.tokenizer
        history_backup = self.history
        strategy_backup = self.strategy
        
        self.model = []
        self.tokenizer = []
        self.history = []
        self.strategy = []
        
        dump(self, os.path.join(save_path, 'model.joblib'))
        
        self.model = model_backup
        self.tokenizer = tokenizer_backup
        self.history = history_backup
        self.strategy = strategy_backup
        
    def load(self, name, parallel_gpu=False):
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
        self = load(os.path.join(model_path, 'model.joblib'))
        
        #tf.distribute.MirroredStrategy is not saved by joblib
        #so we need to update it here
        if parallel_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        #Re-building the model and loading the weights which has been saved
        # in model_path
        self.model, self.tokenizer, self.preprocessing_function = self._getmodel(name)




class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    MetaClassifier(base_estimators, method='voting', from_trained=None, **kwargs)
    
    A wrapper that supports various traditional ensemble classifier (voting and stacking for now),
    following the scikit-learn estimator interface.

    Constructor Arguments:
    * base_estimators: list of tuples with the base estimators ([('name', classifier),...]). 
      To exploit the save and load methods, the base estimators should be instances of class 
      TFbertClassifier, MLClassifier, ImgClassifier or TFmultiClassifier. For boosting and 
      bagging methods only the first estimator of the list will be considered.
    * method (str, optional): Ensemble classifier method. Default is 'voting'.
    * from_trained (optional): Path to previously saved model. Default is None.
    * **kwargs: arguments accepted by the chosen sklearn ensemble classifier sepcified by
        method
    
    Methods:
    * fit(X, y): Trains the model on the provided dataset.
    * predict(X): Predicts the class labels for the given input.
    * predict_proba(X): Predicts class probabilities for the given input (if predict_proba
      is available for the chosen classifier).
    * classification_score(X, y): Calculates weigthed f1-score for the given input and labels.
    * save(name): Saves the model to the directory specified in config.path_to_models.
    
    Example usage:
    meta_classifier = MetaClassifier(base_name='logisticregression', tok_method = 'tfidf')
    meta_classifier.fit(train_texts, train_labels)
    predictions = meta_classifier.predict(test_texts)
    f1score = meta_classifier.classification_score(test_texts, test_labels)
    meta_classifier.save('my_ensemble_model')
        
    """
    def __init__(self, base_estimators, method='voting', from_trained=None, **kwargs):
        """
        Constructor: __init__(self, base_estimators, method='voting', from_trained=None, **kwargs)
        Initializes a new instance of the MetaClassifier.

        Arguments:
        * base_estimators: list of tuples with the base estimators ([('name', classifier),...]). 
          To exploit the save and load methods, the base estimators should be instances of class 
          TFbertClassifier, MLClassifier, ImgClassifier or TFmultiClassifier
        * method (str, optional): Ensemble classifier method. Default is 'voting'.
        * from_trained (optional): Path to previously saved model. Default is None.
        * **kwargs: arguments accepted by the chosen sklearn ensemble classifier sepcified by
            method
        
        Functionality:
        Initializes the classifier based on the specified method and base_estimators and prepares the model 
        for training or inference as specified.
        """
        self.method = method
        self.from_trained = from_trained
        self.base_estimators = base_estimators
        
        if self.from_trained is not None:
            #loading previously saved model if provided
            self.load(self.from_trained)
            self.is_fitted_ = True
        else:
            # Initialize the model according to base_name and kwargs
            if method.lower() == 'voting':
                self.model = VotingClassifier(base_estimators, **kwargs)
            elif method.lower() == 'stacking':
                self.model = StackingClassifier(base_estimators, **kwargs)
            elif method.lower() == 'bagging':
                self.model = BaggingClassifier(estimator=base_estimators[0][1], **kwargs)
            elif method.lower() == 'boosting':
                self.model = AdaBoostClassifier(estimator=base_estimators[0][1], **kwargs)
                
            model_params = self.model.get_params()
            for param, value in model_params.items():
                setattr(self, param, value)
        
        #Only make predict_proba available if self.model 
        # has such method implemented
        if hasattr(self.model, 'predict_proba'):
            self.predict_proba = self._predict_proba
        
        
        
    def fit(self, X, y):
        """
        Trains the model on the provided dataset.

        Arguments:
        * X: The input data used for training. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"
        * y: The target labels for training.
        
        Returns:
        * The instance of MLClassifier after training.
        """
        
        self.model.fit(X, y)
        self.classes_ = np.unique(y)    
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Arguments:
       * X: The input data to use for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"
        
        Returns:
        An array of predicted class labels.
        """
        pred = self.model.predict(X)
        return pred
    
    
    def _predict_proba(self, X):
        """
        Predicts class probabilities for the given text data, if the underlying 
        model supports probability predictions.

        Arguments:
        * X: The input data to use for prediction. Can be an array like a pandas series, 
          or a dataframe with text in column "tokens" and/or image paths in column
          "img_path"
        
        Returns:
        An array of class probabilities for each input instance.
        """
        probs = self.model.predict_proba(X)
        
        return probs
    
    def classification_score(self, X, y):
        """
        Computes scores for the given input X and class labels y
        
        Arguments:
        * X: The text data for which to predict classes. Can be an array like 
          a pandas series, or a dataframe with text in column "tokens" and/or 
          image paths in column "img_path"
        * y: The target labels to predict.
        
        Returns:
        The average weighted f1-score. Also save scores in classification_results
        and f1score attributes
        """
        
        #predict class labels for the input text X
        pred = self.predict(X)
        
        #Save classification report
        self.classification_results = classification_report(y, pred)
        
        #Save weighted f1-score
        self.f1score = f1_score(y, pred, average='weighted')
        
        return self.f1score
    
    def save(self, name):
        """
        Saves the model to the directory specified in src.config file (config.path_to_models).

        Arguments:
        * name: The name to be used for saving the model in config.path_to_models.
        """
        #path to the directory where the model will be saved
        save_path = os.path.join(config.path_to_models, 'trained_models', name)
        
        #Creating it if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        #First saving all base estimators in subfolders
        for k, clf in enumerate(self.estimators_):
            if isinstance(clf, tuple):
                clf[1].save(name + os.sep + clf[0])
            else:
                clf.save(name + os.sep + str(k))
        
        #Removing base estimators to save the meta classifier alone (this is 
        # because joblib does not serialize keras objects)
        estimators_backup = self.estimators_
        if isinstance(self.estimators_[0], tuple):
            for k in range(len(self.estimators_)):
                self.estimators_[k][1] = None
        else:
            self.estimators_ = list(range(len(self.estimators_)))
        
        #Saving meta classifier to that location
        dump(self, os.path.join(save_path, 'model.joblib'))
        
        #Restoring back the base estimators
        self.estimators_ = estimators_backup
        
        
    def load(self, name):
        """
        Loads a model from the directory specified in src.config file (config.path_to_models).

        Arguments:
        * name: The name of the saved model to load.
        """
        #path to the directory where the model to load was saved
        model_path = os.path.join(config.path_to_models, 'trained_models', name)
        
        #Loading the full model from there
        self = load(os.path.join(model_path, 'model.joblib'))
        
        #Loading all base estimators from the  subfolders
        if isinstance(self.estimators_[0], tuple):
            for k, clf in enumerate(self.estimators_):
                base_model = load(os.path.join(model_path, clf[0], 'model.joblib'))
                base_model.load(name + os.sep + clf[0])
                self.estimators_[k] = (clf[0], base_model)
        else:
            for k in range(len(self.estimators_)):
                base_model = load(os.path.join(model_path, str(k), 'model.joblib'))
                base_model.load(name + os.sep + str(k))
                self.estimators_[k] = base_model