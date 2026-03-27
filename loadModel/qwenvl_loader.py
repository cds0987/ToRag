try:
   from unsloth import FastVisionModel
   loadFastVisionModel = 1
except:
    loadFastVisionModel = 0
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base_loader import BaseLoader
from ToRag.encode.qwenVL import qwenVLModel

import logging
logger = logging.getLogger(__name__)
from pprint import pformat
from typing import List, Union, Optional

class QwenVLEncodeLoader(BaseLoader):
    def _load_unsloth(self, model_name: str, **kwargs):

        kwargs["model_name"] = model_name
        load_in_4bit = kwargs.get("load_in_4bit", False)

        if load_in_4bit:
            kwargs["load_in_16bit"] = False

        model, tokenizer = FastVisionModel.from_pretrained(**kwargs)
        return qwenVLModel(model, tokenizer)
    def _load_transformers(self, model_name: str, **kwargs):

        kwargs["model_name"] = model_name
        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        return qwenVLModel(model, processor)
    def load(self, model_name: str, **kwargs):
        loadmode = kwargs.get("loadmode", "default")
        if loadFastVisionModel == 0 and loadmode == "unsloth":
            print("Unsloth library not found, falling back to default transformers loading.")
            loadmode = "default"
        # Remove loadtype from kwargs to avoid passing it to the model loading functions
        kwargs.pop("loadmode", None)
        if loadmode == "unsloth":
            return self._load_unsloth(model_name, **kwargs)

        elif loadmode == "default":
            return self._load_transformers(model_name, **kwargs)
        
# Register the loader
from .load import loader_manager
loader_manager.register_(QwenVLEncodeLoader(), name="qwenvl", type="encode")