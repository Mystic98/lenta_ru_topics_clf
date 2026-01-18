import lenta_topic.preprocess as preprocess
import lenta_topic.tokenize as tokenize
import pandas as pd
import timeit


def main() -> None:
    nlp = preprocess.load_spacy()
    df = pd.read_csv("../data/processed/dataset.csv")
    t0 = timeit.default_timer()
    df['cleaned_text'] = preprocess.spacy_clean_texts(texts=df['text'], nlp=nlp)
    t1 = timeit.default_timer()
    print(f"Cleaning time: {t1 - t0:.2f} seconds")
    df['tokenized_text'] = df['cleaned_text'].apply(tokenize.tokenize)
    df.to_csv("../data/processed/dataset.csv", index=False)
    df['tokenized_text'].to_csv("../data/processed/tokens.csv")
    print('Tokenization complete. Tokenized dataset saved to "../data/processed/tokens.csv".')
    print('Cleaned dataset saved to "../data/processed/dataset.csv".')


if __name__ == '__main__':
    main()


