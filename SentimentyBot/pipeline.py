import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)



def data_prepartion(df, col, tf_id):
    """
    Verileri ön işleme ve TF-IDF dönüşümü yapar.
    
    Argümanlar:
    df (DataFrame): Eğitim veri seti.
    col (str): Metin sütununun adı.
    tf_id (TfidfVectorizer): TF-IDF vektörleştirici.
    
    Dönüş:
    X (sparse matrix): TF-IDF dönüştürülmüş metinler.
    y (ndarray): Etiketler.
    """
    df[col] = df[col].str.lower()

    df['label'] = np.where(df['label'] == 1, 'pozitif', 
                        np.where(df['label'] == -1, 'negatif', 'nötr'))
    
    df["label"] = LabelEncoder().fit_transform(df["label"])
    df = df.dropna()

    X = tf_id.fit_transform(df["tweet"])
    y = df["label"]

    return X, y


def logistic_reg(X, y):
    """
    Lojistik regresyon modeli oluşturur ve çapraz doğrulama yapar.
    
    Argümanlar:
    X (sparse matrix): Özellik matrisi.
    y (ndarray): Etiketler.
    
    Dönüş:
    log_model (LogisticRegression): Eğitimli lojistik regresyon modeli.
    """
    log_model = LogisticRegression().fit(X, y)
    print(cross_val_score(log_model, X, y, scoring="accuracy", cv=5).mean())
    return log_model


def tweets_21(dataframe_new, tweets):
    """
    Yeni tweet veri setindeki metinleri ön işler.
    
    Argümanlar:
    dataframe_new (DataFrame): Yeni tweet veri seti.
    tweets (str): Metin sütununun adı.
    
    Dönüş:
    dataframe_new (DataFrame): Ön işlenmiş veri seti.
    """
    dataframe_new[tweets] = dataframe_new[tweets].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return dataframe_new


def predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer):
    """
    Yeni tweet veri setine tahmin uygular.
    
    Argümanlar:
    dataframe_new (DataFrame): Yeni tweet veri seti.
    log_model (LogisticRegression): Eğitimli lojistik regresyon modeli.
    tf_idfVectorizer (TfidfVectorizer): TF-IDF vektörleştirici.
    
    Dönüş:
    dataframe_new (DataFrame): Etiketlenmiş veri seti.
    """
    tweet_tfidf = tf_idfVectorizer.transform(dataframe_new["tweet"])
    predictions = log_model.predict(tweet_tfidf)
    dataframe_new["label"] = predictions
    return dataframe_new


def main():
    """
    Veri setini yükler, modeli eğitir ve yeni tweetler üzerinde tahmin yapar.
    """
    dataframe = pd.read_csv("tweets_labeled.csv")
    tf_idfVectorizer = TfidfVectorizer()
    X, y = data_prepartion(dataframe, "tweet", tf_idfVectorizer)
    log_model = logistic_reg(X, y)
    dataframe_new = pd.read_csv("tweets_21.csv")
    predicted_df = predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer)


if __name__ == "__main__":
    print("İşlem başladı.")
    main()
