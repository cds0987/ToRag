from abc import ABC, abstractmethod

class encodeModel(ABC):
    def __init__(self, model,*args, **kwargs):
        """
        model: model name or model instance
        kwargs: additional args passed to model loading function
        """
        self.model = model
    @abstractmethod
    def textencode(self,*args, **kwargs):
        pass