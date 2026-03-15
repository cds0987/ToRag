from pprint import pprint
class ModelLoaderManager:

    def __init__(self):
        self.model = {}

    def register_encode(self, loader, name=None):
        if self.model.get("encode") is None:
            self.model["encode"] = {}
        if name:
            self.model["encode"][name] = loader
        else:
            # Use the loader's class name as the default name
            self.model["encode"][loader.__class__.__name__.lower()] = loader

    def loadModel(self,**kwargs):
        print("\n=== Model Configuration ===")
        pprint(kwargs)
        print("=============================\n")
        loadtype = kwargs.get("loadtype", "encode")
        #pop the loadtype from kwargs to avoid passing it to the loaders
        kwargs.pop("loadtype", None)
        modeltype = kwargs.get('modeltype')
        #pop the modeltype from kwargs to avoid passing it to the loaders
        kwargs.pop('modeltype', None)
        loader = self.model[loadtype].get(modeltype)
        if loader:
            return loader.load(**kwargs)
        return None
 
 # GLOBAL INSTANCE
loader_manager = ModelLoaderManager()       