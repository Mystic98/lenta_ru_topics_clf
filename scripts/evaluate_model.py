import lenta_topic.evaluate as evaluate
import lenta_topic.data as data_load
import numpy as np
import pandas as pd

vect_name = input('Enter vectorizer name ("count", "tfidf", "w2v", "w2v_weighted", "navec", "navec_weighted"):')
pipeline = evaluate.load_model(f"../models/pipeline_{vect_name}.joblib")
label_map = evaluate.load_label_map("../models/label_map.json")
topic_map = evaluate.invert_label_map(label_map)
df = pd.read_csv("../data/processed/test_dataset.csv")
_, _, y_pred, _ = evaluate.evaluate_and_save(pipeline, vect_name, df['text'], df['label'], label_names=topic_map)
df['true_topic'] = df['label'].apply(lambda x: topic_map[x])
df['predicted_topic'] = pd.Series(y_pred).apply(lambda x: topic_map[x])
df.to_csv(f"../data/processed/test_dataset_with_predicitons_{vect_name}.csv", index=False)