from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from lenta_topic.vectorizer import Custom_Vectorizer
from lenta_topic.config import *
import joblib


def train_pipeline(X: list[str], y: list[str], vect_name: str, random_state: int = RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    if vect_name == "count":
        vectorizer = CountVectorizer(
            token_pattern=CYR_TOKEN_PATTERN,
            ngram_range=COUNT_NGRAMM_RANGE,
            min_df=COUNT_MIN_DF,
            max_df=COUNT_MAX_DF
        )
    elif vect_name == "tfidf":
        vectorizer = TfidfVectorizer(
            token_pattern=CYR_TOKEN_PATTERN,
            ngram_range=COUNT_NGRAMM_RANGE,
            min_df=COUNT_MIN_DF,
            max_df=COUNT_MAX_DF
        )
    elif vect_name == "w2v":
        vectorizer = Custom_Vectorizer(vectorizer="w2v")

    elif vect_name == "w2v_weighted":
        vectorizer = Custom_Vectorizer(vectorizer= "w2v", tfidf_weighting=True)

    elif vect_name == "navec":
        vectorizer = Custom_Vectorizer(vectorizer="navec")

    elif vect_name == "navec_weighted":
        vectorizer = Custom_Vectorizer(vectorizer= "navec", tfidf_weighting=True)

    else:
        raise ValueError(f"Unknown vectorizer {vect_name}")

    pipe = Pipeline([
        ("vec", vectorizer),
        ("clf", LogisticRegression(C=CLF_C, max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    return pipe, X_train, y_train, X_test, y_test

def save_model(model, path: str):
    joblib.dump(model, path)