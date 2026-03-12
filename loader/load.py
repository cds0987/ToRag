class LoaderManager:

    def __init__(self):
        self.loaders = []

    def register_loader(self, loader):
        self.loaders.append(loader)

    def load(self, folder, workers=8):

        for loader in self.loaders:

            yield from loader.load(folder=folder, workers=workers)
            
            
# GLOBAL INSTANCE
loader_manager = LoaderManager()