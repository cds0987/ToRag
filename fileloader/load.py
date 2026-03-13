class LoaderManager:

    def __init__(self):
        self.file_loaders = []
        self.hf_loaders = []

    def register_file_loader(self, loader):
        self.file_loaders.append(loader)

    def register_hf_loader(self, loader):
        self.hf_loaders.append(loader)

    def load_folder(self, folder, workers=8):

        for loader in self.file_loaders:
            yield from loader.load(folder=folder, workers=workers)

    def load_hfdataset(self, dataset_name, **kwargs):

        for loader in self.hf_loaders:
            yield from loader.load(dataset_name=dataset_name, **kwargs)
            
# GLOBAL INSTANCE
loader_manager = LoaderManager()