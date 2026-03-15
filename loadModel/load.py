from pprint import pprint
class ModelLoaderManager:

    def __init__(self):
        self.ebdloaders = {}

    def register_encode(self, loader, name=None):
        if name:
            self.ebdloaders[name] = loader
        else:
            # Use the loader's class name as the default name
            self.ebdloaders[loader.__class__.__name__.lower()] = loader

    def loadModel(self,**kwargs):
        print("\n=== Model Configuration ===")
        pprint(kwargs)
        print("=============================\n")
        modeltype = kwargs.get('modeltype')
        #pop the modeltype from kwargs to avoid passing it to the loaders
        kwargs.pop('modeltype', None)
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