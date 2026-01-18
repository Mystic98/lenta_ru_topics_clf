import lenta_topic.data as data_load
import pandas as pd

def main() -> None:
    full_df = data_load.load_raw_data('../data/raw/lenta-ru-news.csv.gz')
    #print(full_df['topic'].value_counts(normalize=False))
    #df = data_load.sample_topics_uniform(data_load.filter_topics(full_df))
    df = data_load.data_sample(data_load.filter_topics(full_df))
    #print(df['topic'].value_counts(normalize=False))
    label_map = data_load.make_label_map(df)
    df = data_load.apply_label_map(df, label_map)
    data_load.save_label_map(label_map, "../models/label_map.json")
    df.to_csv("../data/processed/dataset.csv", index=False)
    print(f"Loaded {full_df.shape[0]} records of news")
    print(f"Saved processed dataset: ../data/processed/dataset.csv")
    print(f"Saved label map: ../models/label_map.json")



if __name__ == '__main__':
    main()

