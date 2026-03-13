from typing import List, Union, Optional
import numpy as np
from llama_cpp import Llama
from .base_loader import BaseLoader
from ToRag.encode.llamacpp import llamacppModel
class llamacppLoader(BaseLoader):

    def load(self, **kwargs):
        model = Llama(**kwargs)
        return llamacppModel(model, **kwargs)
    
# Register the loader
from .load import loader_manager
loader_manager.register_llamacppebd_loader(llamacppLoader())