from unsloth import FastSentenceTransformer
from sentence_transformers import SentenceTransformer
from .base_loader import BaseLoader
import logging
logger = logging.getLogger(__name__)
from pprint import pformat
from typing import List, Union, Optional
from ToRag.encode.sentencetransformer import sentencetransformermodel
class sentencetransformerLoader(BaseLoader):

    def load(self, model_name: str, **kwargs):
        logger.info("Loading encoder with configuration:\n%s", pformat(kwargs))
        loadtype = kwargs.get("loadtype", "default")
        # Remove loadtype from kwargs to avoid passing it to the model loading functions
        kwargs.pop("loadtype", None)
        if loadtype == "unsloth":
            return self._load_unsloth(model_name, **kwargs)

        elif loadtype == "default":
            return self._load_sentence_transformer(model_name, **kwargs)
    def _load_unsloth(self, model_name: str, **kwargs):

        kwargs["model_name"] = model_name
        load_in_4bit = kwargs.get("load_in_4bit", False)

        if load_in_4bit:
            kwargs["load_in_16bit"] = False

        model = FastSentenceTransformer.from_pretrained(**kwargs)
        return sentencetransformermodel(model, **kwargs)

    def _load_sentence_transformer(self, model_name: str, **kwargs):

        kwargs["model_name"] = model_name
        model = SentenceTransformer(**kwargs)
        return  sentencetransformermodel(model, **kwargs)

# Register the loader
from .load import loader_manager
loader_manager.register_sentencetransformer_loader(sentencetransformerLoader())