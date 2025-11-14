from sklearn.cluster import KMeans
import joblib

class CandleClusterer:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=2)

    def train(self, X):
        self.model.fit(X)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)