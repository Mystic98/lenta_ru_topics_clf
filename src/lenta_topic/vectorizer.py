import re
import numpy as np
import gensim
from navec import Navec
from gensim.corpora import Dictionary
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from lenta_topic.config import *


class Custom_Vectorizer(BaseEstimator, TransformerMixin):
    """
    Turn a document (list[str]) into a dense vector by averaging word vectors.

    Modes:
      - vectorizer='w2v': train Word2Vec on provided corpus
      - vectorizer='navec': load pretrained Navec vectors from navec_path

    Optionally uses TF-IDF weights (trained on the same tokenized corpus) for a weighted average.

    Input X must be: list[str]
    Output: np.ndarray shape (n_samples, vector_size)
    """

    def __init__(
        self,
        vector_size: int = W2V_VECTOR_SIZE,
        window: int = W2V_WINDOW,
        min_count: int = W2V_MIN_COUNT,
        workers: int = W2V_WORKERS,
        epochs: int = W2V_EPOCHS,
        seed: int = W2V_SEED,
        sg: int = W2V_SG,
        negative: int = W2V_NEGATIVE,
        sample: float = W2V_SAMPLE,
        alpha: float = W2V_ALPHA,
        min_alpha: float = W2V_MIN_ALPHA,
        vectorizer: str = "w2v",              # 'w2v' or 'navec'
        tfidf_weighting: bool = W2V_TFIDF_WEIGHTING,
        navec_path: str = NAVEC_PATH,         # path to navec .tar (relative or absolute)
        navec_prefix: str = "../",            # keeps your old behavior; set "" if not needed
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.sg = sg
        self.negative = negative
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha

        self.vectorizer = vectorizer
        self.tfidf_weighting = tfidf_weighting
        self.navec_path = navec_path
        self.navec_prefix = navec_prefix

    def fit(self, X, y=None):
        if not isinstance(X, (list, tuple)):
            X = list(X)

        # Tokenize once for both modes
        tokenized_X = [re.findall(CYR_TOKEN_PATTERN, str(text).lower()) for text in X]

        # Always build Dictionary on corpus tokens (needed for TF-IDF & stable token->id)
        self.dct_ = Dictionary(tokenized_X)

        if self.vectorizer == "w2v":
            self.model_ = gensim.models.Word2Vec(
                sentences=tokenized_X,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                epochs=self.epochs,
                seed=self.seed,
                sg=self.sg,
                negative=self.negative,
                sample=self.sample,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
            ).wv

        elif self.vectorizer == "navec":
            # Load pretrained Navec vectors
            self.model_ = Navec.load(f"{self.navec_prefix}{self.navec_path}")

            # Align vector_size with navec dimensionality
            try:
                first_word = self.model_.vocab.words[0]
                self.vector_size = int(self.model_[first_word].shape[0])
            except Exception:
                # If dimension can't be inferred, keep configured vector_size
                pass

        else:
            raise ValueError("vectorizer must be either 'w2v' or 'navec'")

        # Fit TF-IDF if requested (works for both w2v and navec)
        if self.tfidf_weighting:
            corpus_bow = [self.dct_.doc2bow(doc) for doc in tokenized_X]
            self.tfidf_model_ = gensim.models.TfidfModel(corpus_bow)

        self.n_samples_seen_ = len(tokenized_X)
        return self

    def transform(self, X):
        check_is_fitted(self, ["model_", "dct_"])

        if not isinstance(X, (list, tuple)):
            X = list(X)

        tokenized_X = [re.findall(CYR_TOKEN_PATTERN, str(text).lower()) for text in X]
        wv = self.model_
        dim = self.vector_size

        if not tokenized_X:
            return np.empty((0, dim), dtype=np.float32)

        # Unified access for gensim KeyedVectors vs Navec
        def has_vec(word: str) -> bool:
            if self.vectorizer == "w2v":
                return word in wv.key_to_index
            return word in wv  # Navec supports membership test

        def get_vec(word: str) -> np.ndarray:
            return wv[word]

        def mean_docvec(tokens):
            vecs = [get_vec(w) for w in tokens if has_vec(w)]
            if not vecs:
                return np.zeros(dim, dtype=np.float32)
            return np.mean(vecs, axis=0).astype(np.float32)

        def tfidf_docvec(tokens):
            bow = self.dct_.doc2bow(tokens)

            vec = np.zeros(dim, dtype=np.float32)
            wsum = 0.0

            for token_id, weight in self.tfidf_model_[bow]:
                word = self.dct_[token_id]
                if has_vec(word):
                    vec += weight * get_vec(word)
                    wsum += weight

            if wsum > 0:
                vec /= wsum
            return vec.astype(np.float32)

        if self.tfidf_weighting:
            check_is_fitted(self, ["tfidf_model_"])
            X_out = np.vstack([tfidf_docvec(doc) for doc in tokenized_X])
        else:
            X_out = np.vstack([mean_docvec(doc) for doc in tokenized_X])

        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"emb_{i}" for i in range(self.vector_size)])