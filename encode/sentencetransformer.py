import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
#Load from Unsloth
from unsloth import FastSentenceTransformer

def loadUnsloth(model_name: str, **kwargs):
    logger.info(f"Loading Unsloth FastSentenceTransformer model: {model_name}")
    kwargs["model_name"] = model_name
    load_in_4bit = kwargs.get("load_in_4bit", False)
    if load_in_4bit:
        kwargs['load_in_16bit'] = False
    model = FastSentenceTransformer.from_pretrained(**kwargs)
    return model.eval()


from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
def load_sentence_transformer(model_name: str, **kwargs) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    Args:
        model_name: name of the model to load
        device: device to load the model on (cpu / cuda / mps)
    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    kwargs["model_name"] = model_name
    model = SentenceTransformer(**kwargs)
    return model.eval()


from pprint import pformat
def loadencoder(modelname: str,loadtype,**kwargs):
    logger.info("Loading encoder with configuration:\n%s", pformat(kwargs))
    if loadtype == "unsloth":
        return loadUnsloth(modelname, **kwargs)
    elif loadtype == "sentence_transformer":
        return load_sentence_transformer(modelname, **kwargs)
    else:
        raise ValueError(f"Unsupported load type: {loadtype}")



from .model import encodeModel
class textencode(encodeModel):
    def __init__(
        self,
        model_name: str, **kwargs
    ):
        """
        Professional text embedding wrapper.
        Args:
            model: model name or SentenceTransformer instance
            device: cpu / cuda / mps
        """

        super().__init__(model_name, **kwargs)
        self.model = loadencoder(model_name, **kwargs)
    def encode(
        self,
        documents: Union[str, List[str]],
        **kwargs,
    ) -> np.ndarray:
        """
        Encode documents into embeddings.
        Args:
            documents: text or list of texts
            **kwargs: additional args passed to model.encode(),similar with Sentence Transformer library
        """
        embeddings = None
        if isinstance(documents, str):
            documents = [documents]
        try:
          with torch.no_grad():
            embeddings = self.model.encode(
            documents,
            convert_to_numpy = True,
            **kwargs
        )
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            pass
        torch.cuda.empty_cache()
        return embeddings
    
