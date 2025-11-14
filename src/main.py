from .utils.config import CONFIG
from .clustering.preprocessing import CandlePreprocessor
from .clustering.clusterer import CandleClusterer

def main():

    # load and preprocess
    pre = CandlePreprocessor(CONFIG['data'])
    df = pre.load_data("data/raw/data.csv")
    df = pre.organize_datetime(df)
    X = pre.generate_features(df)

    # train clusters
    clusterer = CandleClusterer(n_clusters=CONFIG['clustering']['n_clusters'])
    clusterer.train(X)
    labels = clusterer.predict(X)

if __name__ == '__main__':
    main()