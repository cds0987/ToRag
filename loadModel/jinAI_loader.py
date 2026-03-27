from ToRag.encode.jinAI import jinAIModel
from .base_loader import BaseLoader
from ToRag.model.jinai import Jinai

class jinAIEncodeLoader(BaseLoader):

    def load(self, model_name: str, **kwargs):
        kwargs["model_name"] = model_name
        model =  Jinai(**kwargs)
        return jinAIModel(model)
    
    
# Register the loader
from .load import loader_manager
loader_manager.register_(jinAIEncodeLoader(), name="jinai", type="encode")