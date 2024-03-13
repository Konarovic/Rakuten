from gensim.models import Word2Vec
from .word2vec_vectorizer import Word2VecVectorizer


class SkipGramVectorizer(Word2VecVectorizer):
    """
    SkipGramVectorizer

    This class is a transformer that uses the SkipGram Word2Vec model to transform a list of sentences into a list of vectors.

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
        Fit the Word2Vec model to the sentences in X.

    transform(X)
        Transform the sentences in X into vectors using the Word2Vec model.

    vectorize(sentence)
        Transform a sentence into a vector using the Word2Vec model.


    Example
    -------
    >>> from skipgram_vectorizer import SkipGramVectorizer
    >>> sentences = ["this is a sentence", "this is another sentence"]
    >>> vectorizer = SkipGramVectorizer()
    >>> vectorizer.fit(sentences)
    >>> vectors = vectorizer.transform(sentences)
    >>> print(vectors)
    """

    def fit(self, X, y=None):
        sentences = []
        for sentence in X:
            if not isinstance(sentence, str):
                sentences.append(sentence)
                raise (
                    "All sentences should be strings : line {} - {}".format(len(sentences), sentence))
            else:
                sentences.append(sentence.split())

        self.w2v_model = Word2Vec(
            sentences,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            vector_size=self.vector_size,
            sg=1
        )
        return self
