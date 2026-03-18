import faiss
from faissDev.base import FaissIndex

class IndexLSH(FaissIndex):

    def __init__(
        self,
        dimension=384,
        nbits=256,
        vectornormalize=True
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nbits = nbits

        self._create_index()

    # -------------------------
    # create index
    # -------------------------
    def _create_index(self):

        # LSH ignores metric (always binary hashing)
        base = faiss.IndexLSH(
            self.dimension,
            self.nbits
        )

        # wrap for ID support
        self.index = faiss.IndexIDMap2(base)