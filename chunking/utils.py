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
