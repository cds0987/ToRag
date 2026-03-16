from typing import List, Union, Optional
import numpy as np
try:
  from llama_cpp import Llama
except ImportError:
  print("llama_cpp library is required for llamacppLoader. Please install it with `pip install llama-cpp-python` or never use llamacppLoader else unexpected behavior.")
  Llama = None
from .base_loader import BaseLoader

class llamacppEncodeLoader(BaseLoader):

    def load(self, **kwargs):
        model = Llama.from_pretrained(**kwargs)
        from ToRag.encode.llamacpp import llamacppModel
        return llamacppModel(model, **kwargs)
    
# Register the loader
from .load import loader_manager
loader_manager.register_(llamacppEncodeLoader(), name="llamacpp", type="encode")