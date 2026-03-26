from abc import ABC, abstractmethod

class encodeModel(ABC):
    def __init__(self, model,*args, **kwargs):
        """
        model: model name or model instance
        kwargs: additional args passed to model loading function
        """
        self.model = model
    def textencode(self,*args, **kwargs):
        pass
    def imgencode(self,*args, **kwargs):
        pass
    def encode(self,*args, **kwargs):
        pass