from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseDataManager(ABC):
    """
    Abstract base class for managing document retrieval in a RAG (Retrieval-Augmented Generation) system.

    This class defines a standardized interface and shared logic for:
    - Mapping document IDs to indices
    - Fetching documents in batches
    - Formatting outputs into multiple representations
    - Reconstructing query-wise Top-K results

    Subclasses are responsible for implementing backend-specific data access logic.

    Design Principles:
    - Template Method Pattern: `get_documents` defines workflow, subclasses implement `_fetch_subset`
    - Separation of Concerns: retrieval vs formatting vs indexing
    - Extensibility: easy to plug in new data sources (DB, API, vector store)

    Attributes:
        id_field (str): Field name used as unique document identifier.
        text_fields (List[str]): Fields used to construct document content.
        title_field (Optional[str]): Optional field for document titles.
        id2idx (Dict[str, int]): Mapping from document ID to index.
        idx2id (Dict[int, str]): Mapping from index to document ID.
    """

    def __init__(
        self, 
        id_field: str = "_id", 
        text_fields: Union[str, List[str]] = "text", 
        title_field: Optional[str] = None
    ):
        """
        Initialize the BaseDataManager.

        Args:
            id_field (str): Name of the unique identifier field in each record.
            text_fields (Union[str, List[str]]): Field(s) used to construct document content.
            title_field (Optional[str]): Optional title field for documents.
        """
        self.id_field = id_field
        self.text_fields = [text_fields] if isinstance(text_fields, str) else text_fields
        self.title_field = title_field
        self.id2idx: Dict[str, int] = {}
        self.idx2id: Dict[int, str] = {}

    @abstractmethod
    def __len__(self) -> int:
        """
        Return total number of records in the dataset.

        Must be implemented by subclasses.

        Returns:
            int: Total number of documents.
        """
        pass

    @abstractmethod
    def _fetch_subset(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch a subset of documents based on provided indices.

        This method must be implemented by subclasses and should contain
        backend-specific logic for retrieving data.

        Args:
            indices (List[int]): List of integer indices to retrieve.

        Returns:
            List[Dict[str, Any]]: List of document records.
        """
        pass

    def _build_mappings(self, ids: List[Any]):
        """
        Build bidirectional mappings between document IDs and indices.

        This enables fast lookup during retrieval and alignment with external retrievers.

        Args:
            ids (List[Any]): List of document IDs.
        """
        self.id2idx = {str(id_): i for i, id_ in enumerate(ids)}
        self.idx2id = {i: str(id_) for i, id_ in enumerate(ids)}

    # --- Shared Retrieval Logic ---

    def get_documents(
        self, 
        indices: np.ndarray, 
        scores: Optional[np.ndarray] = None, 
        return_type: str = "dict"
    ) -> List[List[Union[Dict, str]]]:
        """
        Retrieve and format documents based on index matrix (e.g., Top-K results).

        This is the main public API used in RAG pipelines.

        Workflow:
            1. Flatten indices
            2. Filter invalid indices (-1 or out of bounds)
            3. Fetch subset via `_fetch_subset`
            4. Format documents
            5. Reconstruct original batch structure

        Args:
            indices (np.ndarray): 2D array of shape (batch_size, top_k).
            scores (Optional[np.ndarray]): Optional relevance scores aligned with indices.
            return_type (str): Output format ("dict", "text", "raw").

        Returns:
            List[List[Union[Dict, str]]]: Nested list where each inner list contains
            Top-K documents for a query.
        """
        indices = np.asarray(indices)
        flat_indices = indices.flatten()
        
        # Filter valid indices
        valid_mask = (flat_indices != -1) & (flat_indices < len(self))
        valid_indices = flat_indices[valid_mask].tolist()

        if not valid_indices:
            return [[] for _ in range(indices.shape[0])]

        # Fetch using subclass implementation
        subset = self._fetch_subset(valid_indices)
        
        # Apply formatting
        formatted_batch = self._apply_formatting(subset, return_type)
        
        # Reconstruct batch structure
        return self._reconstruct_matrix(indices, scores, formatted_batch)

    # --- Shared Formatting Logic ---

    def _apply_formatting(self, subset: List[Dict], return_type: str) -> List[Any]:
        """
        Apply formatting strategy to retrieved documents.

        Args:
            subset (List[Dict]): Raw documents.
            return_type (str): Desired output format.

        Returns:
            List[Any]: Formatted documents.
        """
        if return_type == "raw": return subset
        if return_type == "text": return self._format_as_text(subset)
        return self._format_as_dict(subset)

    def _format_as_text(self, batch: List[Dict]) -> List[str]:
        """
        Format documents as plain text strings.

        Each document is constructed by concatenating specified text fields.

        Args:
            batch (List[Dict]): Raw documents.

        Returns:
            List[str]: Text-formatted documents.
        """
        return ["\n".join([f"{f}: {item.get(f)}" for f in self.text_fields]) for item in batch]

    def _format_as_dict(self, batch: List[Dict]) -> List[Dict]:
        """
        Format documents into structured dictionary format.

        Structure:
            {
                "id": ...,
                "content": ...,
                "metadata": {...}
            }

        Args:
            batch (List[Dict]): Raw documents.

        Returns:
            List[Dict]: Structured documents.
        """
        return [{
            "id": item.get(self.id_field),
            "content": "\n".join([str(item.get(f)) for f in self.text_fields]),
            "metadata": item
        } for item in batch]

    def _reconstruct_matrix(self, indices, scores, flat_docs) -> List[List[Any]]:
        """
        Reconstruct the original (batch_size, top_k) structure of retrieved documents.

        This method aligns flattened documents back to their query groups and
        attaches scores if provided.

        Notes:
            - Invalid indices (-1 or out-of-range) are skipped
            - Uses pointer-based traversal for efficiency
            - Mutates document objects when attaching scores (be cautious)

        Args:
            indices (np.ndarray): Original index matrix.
            scores (Optional[np.ndarray]): Score matrix.
            flat_docs (List[Any]): Flattened list of documents.

        Returns:
            List[List[Any]]: Reconstructed nested list of documents.
        """
        results = []
        ptr = 0
        for i in range(indices.shape[0]):
            row = []
            for j in range(indices.shape[1]):
                if indices[i, j] == -1 or indices[i, j] >= len(self): continue
                doc = flat_docs[ptr]
                if scores is not None:
                    if isinstance(doc, dict): doc["score"] = float(scores[i][j])
                    else: doc = {"text": doc, "score": float(scores[i][j])}
                row.append(doc)
                ptr += 1
            results.append(row)
        return results
