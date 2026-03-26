try:
   from unsloth import FastSentenceTransformer
   loadFastSentenceTransformer = 1
except:
    loadFastSentenceTransformer = 0
from sentence_transformers import SentenceTransformer
from .base_loader import BaseLoader
import logging
logger = logging.getLogger(__name__)
from pprint import pformat
from typing import List, Union, Optional
from ToRag.encode.sentencetransformer import SentenceTransformerModel
from pprint import pprint

class sentencetransformerLoader(BaseLoader):

    def load(self, model_name: str, **kwargs):
        loadmode = kwargs.get("loadmode", "default")
        if loadFastSentenceTransformer == 0 and loadmode == "unsloth":
            print("Unsloth library not found, falling back to default SentenceTransformer loading.")
            loadmode = "default"
        # Remove loadtype from kwargs to avoid passing it to the model loading functions
        kwargs.pop("loadmode", None)
        if loadmode == "unsloth":
            return self._load_unsloth(model_name, **kwargs)

        elif loadmode == "default":
            return self._load_sentence_transformer(model_name, **kwargs)
    def _load_unsloth(self, model_name: str, **kwargs):

        kwargs["model_name"] = model_name
        load_in_4bit = kwargs.get("load_in_4bit", False)

        if load_in_4bit:
            kwargs["load_in_16bit"] = False

        model = FastSentenceTransformer.from_pretrained(**kwargs)
        return SentenceTransformerModel(model, **kwargs)

    def _load_sentence_transformer(self, model_name: str, **kwargs):

        kwargs["model_name_or_path"] = model_name
        max_seq_length = kwargs.get("max_seq_length", 512)
        #pop max_seq_length from kwargs to avoid passing it to fit sentence transformer library
        kwargs.pop("max_seq_length", None)
        model = SentenceTransformer(**kwargs)
        model.max_seq_length = max_seq_length
        return  SentenceTransformerModel(model, **kwargs)

# Register the loader
from .load import loader_manager
loader_manager.register_(sentencetransformerLoader(), name="sentencetransformer", type="encode")