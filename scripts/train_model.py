import lenta_topic.train as train
import pandas as pd
import timeit
from sklearn.metrics import classification_report


def main() -> None:
    vect_name = input("Enter vectorizer name ('count', 'tfidf', 'w2v', 'w2v_weighted', 'navec', 'navec_weighted'): ")
    if vect_name not in ['count', 'tfidf', 'w2v', 'w2v_weighted', 'navec', 'navec_weighted']:
        raise ValueError("Invalid vectorizer name.")
    df = pd.read_csv("../data/processed/dataset.csv")
    t_0 = timeit.default_timer()
    pipe, X_train, y_train, X_test, y_test = train.train_pipeline(X=df['cleaned_text'], y=df['label'], vect_name=vect_name)
    t_1 = timeit.default_timer()
    print(f"Training time: {t_1 - t_0:.2f} seconds")
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    train_df.to_csv("../data/processed/train_dataset.csv", index=False)
    test_df.to_csv("../data/processed/test_dataset.csv", index=False)
    train.save_model(pipe, path=f'../models/pipeline_{vect_name}.joblib')
    print(f'Model saved to "../models/pipeline_{vect_name}.joblib".')
    print('Train and test datasets saved to "../data/processed/train_dataset.csv" and "../data/processed/test_dataset.csv".')
    #print(classification_report(y_train, pipe.predict(X_train)))

if __name__ == '__main__':
    main()