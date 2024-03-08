import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word2VecVectorizer

    This class is an abstract class transformer that uses the Word2Vec model to transform a list of sentences into a list of vectors.

    CAUTION : should not be instantiated directly, use one of the child classes instead.

    Parameters
    ----------
    vector_size : int, default=500
        The size of the word vectors.

    window : int, default=10
        The maximum distance between the current and predicted word within a sentence.

    min_count : int, default=2
        Ignores all words with total frequency lower than this.

    workers : int, default=4
        The number of workers to use for training the model.


    Attributes
    ----------
    w2v_model : Word2Vec
        The Word2Vec model used to transform the sentences into vectors.    


    Methods
    -------
    fit(X, y=None)
        Fit the Word2Vec model to the sentences in X. Has to be overriden by the child class.

    transform(X)
        Transform the sentences in X into vectors using the Word2Vec model.

    vectorize(sentence)
        Transform a sentence into a vector using the Word2Vec model.
    """

    def __init__(self, vector_size=500, window=10, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.w2v_model = None

    def fit(self, X, y=None):
        raise NotImplementedError(
            "fit method has to be overriden by the child class")
        return self

    def transform(self, X):
        return np.array([self.vectorize(sentence) for sentence in X])

    def vectorize(self, sentence):
        sentence = str(sentence).lower()
        words = sentence.split()
        words_vecs = [self.w2v_model.wv[word]
                      for word in words if word in self.w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(self.w2v_model.wv.vector_size)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)
