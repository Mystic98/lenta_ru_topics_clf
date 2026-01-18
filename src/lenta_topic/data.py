from __future__ import annotations
import json
import pandas as pd
from corus import load_lenta
from numpy.ma.core import size
from lenta_topic.config import *

def load_raw_data(path: str) -> pd.DataFrame:
    records = load_lenta(path)
    rows = [{'text': r.text + ' ' + r.title, 'topic': r.topic} for r in records]
    return pd.DataFrame(rows)


def filter_topics(df: pd.DataFrame, min_count: int = MIN_TOPIC_COUNT) -> pd.DataFrame:
    counts = df["topic"].value_counts()
    keep = counts[counts >= min_count].index
    return df[df["topic"].isin(keep)].copy().reset_index(drop=True)

def data_sample(df: pd.DataFrame, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    return df.sample(size, random_state=RANDOM_STATE, ignore_index=True)


def sample_topics_uniform(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby("topic")
    parts = []
    for topic, group in groups:
        part = group.sample(MIN_TOPIC_COUNT, random_state=RANDOM_STATE)
        parts.append(part)
    sample = pd.concat(parts)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE)
    return sample


def make_label_map(df: pd.DataFrame) -> dict[str, int]:
    topics = sorted(df["topic"].unique())
    return {t: i for i, t in enumerate(topics)}


def apply_label_map(df: pd.DataFrame, label_map: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["topic"].map(label_map)
    return df


def save_label_map(label_map: dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)









