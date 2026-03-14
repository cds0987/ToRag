from typing import List, Union, Optional
import numpy as np
try:
  from llama_cpp import Llama
except ImportError:
  raise ImportError("llama_cpp library is required for llamacppLoader. Please install it with `pip install llama-cpp-python`.")
  Llama = None
from .base_loader import BaseLoader
from ToRag.encode.llamacpp import llamacppModel
class llamacppLoader(BaseLoader):

    def load(self, **kwargs):
        model = Llama.from_pretrained(**kwargs)
        return llamacppModel(model, **kwargs)
    
# Register the loader
from .load import loader_manager
loader_manager.register_llamacppebd_loader(llamacppLoader())