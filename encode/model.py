class encodeModel:
    def __init__(self, modelname,*args, **kwargs):
        """
        model: model name or model instance
        kwargs: additional args passed to model loading function
        """
        self.modelname = modelname
    def encode(self,*args, **kwargs):
        pass