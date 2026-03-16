from ToRag.encode.tokenencode import tokenencodeModel
from .base_loader import BaseLoader
from ToRag.model.jinai import jinaitokenencode

class jinAItokenencodeLoader(BaseLoader):

    def load(self, model_name: str, **kwargs):
        kwargs["model_name"] = model_name
        model =  jinaitokenencode(**kwargs)
        return tokenencodeModel(model)
    
    
# Register the loader
from .load import loader_manager
loader_manager.register_(jinAItokenencodeLoader(), name="jinaitokenencode", type="encode")