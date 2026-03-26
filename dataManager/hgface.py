from .base import BaseDataManager

class HuggingFaceDataManager(BaseDataManager):
    """
    Concrete implementation of BaseDataManager for Hugging Face datasets.

    This class adapts datasets from the Hugging Face ecosystem to the standardized
    retrieval interface defined by BaseDataManager.

    It leverages the dataset's built-in `.select()` method for efficient batch retrieval.

    Expected Dataset Format:
        - Must support indexing via `dataset[field_name]`
        - Must support `.select(indices)` for subset retrieval
        - Must contain the `id_field` specified in BaseDataManager

    Attributes:
        dataset: Hugging Face dataset object
    """

    def __init__(self, dataset, **kwargs):
        """
        Initialize HuggingFaceDataManager.

        Args:
            dataset: Hugging Face dataset instance.
            **kwargs: Additional arguments passed to BaseDataManager.
        """
        super().__init__(**kwargs)
        self.dataset = dataset

        # Build ID-index mappings using dataset column
        self._build_mappings(dataset[self.id_field])

    def __len__(self):
        """
        Return total number of records in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.dataset)

    def _fetch_subset(self, indices):
        """
        Fetch a subset of documents using Hugging Face dataset's select method.

        This method performs efficient row selection and returns records
        as a list of dictionaries.

        Args:
            indices (List[int]): List of indices to retrieve.

        Returns:
            List[Dict]: Retrieved documents.
        """
        return list(self.dataset.select(indices))