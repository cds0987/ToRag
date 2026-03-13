class ModelLoaderManager:

    def __init__(self):
        self.ebdloaders = {}

    def register_sentencetransformer_loader(self, loader):
        self.ebdloaders['sentencetransformer'] = loader

    def register_llamacppebd_loader(self, loader):
        self.ebdloaders['llamacpp'] = loader
    def loadModel(self,**kwargs):
        modeltype = kwargs.get('modeltype')
        if modeltype == 'sentencetransformer':
            loader = self.ebdloaders.get('sentencetransformer')
            if loader:
                return loader.load(**kwargs)
        elif modeltype == 'llamacpp':
            loader = self.ebdloaders.get('llamacpp')
            if loader:
                return loader.load(**kwargs)
        return None
 
 # GLOBAL INSTANCE
loader_manager = ModelLoaderManager()       