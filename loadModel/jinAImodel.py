from ToRag.encode.tokenencode import tokenencode
from .base_loader import BaseLoader
from ToRag.model.jinai import jinaitokenencode

class jinAImodelLoader(BaseLoader):

    def load(self, model_name: str, **kwargs):
        kwargs["model_name"] = model_name
        model =  jinaitokenencode(**kwargs)
        return tokenencode(model)
    
    
# Register the loader
from .load import loader_manager
loader_manager.register_encode(jinAImodelLoader(), name="jinai")