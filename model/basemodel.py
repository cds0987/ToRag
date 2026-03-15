from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        pass
