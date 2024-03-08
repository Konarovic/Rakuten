from gensim.models import Word2Vec
from .word2vec_vectorizer import Word2VecVectorizer


class CBowVectorizer(Word2VecVectorizer):
    """
    CBowVectorizer

    This class is a transformer that uses the CBow Word2Vec model to transform a list of sentences into a list of vectors.

    Parameters
    ----------
    vector_size : int, default=100
        The size of the word vectors.

    window : int, default=5
        The maximum distance between the current and predicted word within a sentence.

    min_count : int, default=5
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
        Fit the Word2Vec model to the sentences in X.

    transform(X)
        Transform the sentences in X into vectors using the Word2Vec model.

    vectorize(sentence)
        Transform a sentence into a vector using the Word2Vec model.


    Example
    -------
    >>> from cbow_vectorizer import CBowVectorizer
    >>> sentences = ["this is a sentence", "this is another sentence"]
    >>> vectorizer = CBowVectorizer()
    >>> vectorizer.fit(sentences)
    >>> vectors = vectorizer.transform(sentences)
    >>> print(vectors)
    """

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]
        self.w2v_model = Word2Vec(
            sentences,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            vector_size=self.vector_size,
            sg=0
        )

        return self
