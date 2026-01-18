from __future__ import annotations
from typing import Iterable
import spacy
from spacy import Language
from lenta_topic.config import *


def load_spacy(model: str = SPACY_MODEL):
    return spacy.load(model, disable=["ner", "parser"])


def spacy_clean_texts(texts: list[str], nlp : Language, batch_size=SPACY_BATCH_SIZE, n_process=SPACY_N_PROCESS):
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop
            and not t.is_punct
            and not t.like_email
            and not t.like_num
            and not t.is_space
        ]
        cleaned.append(' '.join(tokens))
    return cleaned

