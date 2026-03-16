import nltk

def download_nltk_dependencies():
    packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "vader_lexicon",
        "maxent_ne_chunker",
        "words",
        'punkt_tab'
    ]

    for pkg in packages:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg)
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from collections import defaultdict
class ClusterStrategy:

    def __init__(self, model_cls = None, max_chunk_size=5):
        self.model_cls = model_cls if model_cls is not None else KMeans()
        self.max_chunk_size = max_chunk_size

    def set_clusters(self, n_clusters):
        self.model_cls.n_clusters = n_clusters

    def fit_predict(self, embeddings, **kwargs):
        n_clusters =  kwargs.get("n_clusters", max(1, len(embeddings) // self.max_chunk_size))
        self.set_clusters(n_clusters)
        return self.model_cls.fit_predict(embeddings)
