from .Noise import Noise
from .Object import Pointcloud
from .Preprocessor import Preprocessor

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2 as shutil_copy2
from torch import (
    cat as torch_cat,
    load as torch_load,
    randperm as torch_randperm,
    save as torch_save
)
from torch_geometric.data import (
    Batch as tg_data_Batch,
    InMemoryDataset as tg_data_InMemoryDataset
)
from typing import List
from warnings import warn as warnings_warn

@dataclass
class NoiseLevels:
    gaussian: List[float]
    impulsive: List[float]

@dataclass
class DataBalancing:
    feature: float = None
    nonfeature: float = None

    # Make sure the variables sum to 1 and can be set individually.
    def __post_init__(self):
        f_in_range = lambda input, start, end: input >= start and input <= end
        if self.feature is None and self.nonfeature is None:
            raise ValueError("One of the attributes must be set.")
        elif self.feature is None and self.nonfeature is not None:
            if f_in_range(self.nonfeature, 0, 1):
                self.feature = 1 - self.nonfeature
            else:
                raise ValueError("nonfeature must be within 0 and 1")
        elif self.nonfeature is None and self.feature is not None:
            if f_in_range(self.feature, 0, 1):
                self.nonfeature = 1 - self.feature
            else:
                raise ValueError("nonfeature must be within 0 and 1")
        elif 1 - (self.feature + self.nonfeature) >= 0.001:
            raise ValueError("feature and nonfeature groups must sum to 1")

class PatchDataset(tg_data_InMemoryDataset):

    DEFAULT_NOISE_LEVELS = NoiseLevels(
        gaussian = [0.1, 0.2, 0.3],
        impulsive = [0.1, 0.2, 0.3]
    )
    EXTENSION = ".pt"
    CLASSES = "_classes"
    GAUSSIAN = "_gaussian_"
    IMPULSIVE = "_impulsive_"

    def __init__(self, root, objects: List[Pointcloud], transform=None, pre_transform=None, noise_levels: NoiseLevels=DEFAULT_NOISE_LEVELS):
        if len(objects) == 0:
            raise ValueError("Cannot create an empty dataset.")
        self.objects = objects
        self.noise_levels = noise_levels
        super(PatchDataset, self).__init__(root, transform, pre_transform)
        if transform is not None or pre_transform is not None:
            warnings_warn("transform or pre_transform given. Methods for these are not implemented!")
        _processed_paths = self.processed_paths
        graphs = []
        for path in [x for x in _processed_paths if x.find(self.CLASSES) == -1]:
            store = torch_load(path)
            graph_list = tg_data_Batch.to_data_list(store)
            graphs += graph_list
        self.data, self.slices = self.collate(graphs)
    
    @property
    def raw_file_names(self):
        return [Path(x.file_path).name for x in self.objects]

    @property
    def processed_file_names(self):
        _nl = self.noise_levels
        result = []
        for path in self.raw_paths:
            _stem = Path(path).stem
            result.append(_stem + self.CLASSES + self.EXTENSION)
            for level in _nl.impulsive:
                result.append(_stem + self.IMPULSIVE + str(level) + self.EXTENSION)
            for level in _nl.gaussian:
                result.append(_stem + self.GAUSSIAN + str(level) + self.EXTENSION)
        return result

    def download(self):
        for path in [x.file_path for x in self.objects]:
            shutil_copy2(path, self.raw_dir)

    def process(self):
        _raw_paths = self.raw_paths
        _nl = self.noise_levels
        _pdir = Path(self.processed_dir)
        data_list = []
        for path in _raw_paths:
            _stem = Path(path).stem
            pointcloud = Pointcloud(path)
            preprocessor = Preprocessor(pointcloud)
            noise = Noise(pointcloud)
            classes_file = _pdir / (_stem + self.CLASSES + self.EXTENSION)
            if not classes_file.exists():
                classes = preprocessor.getClasses()
                torch_save(classes, classes_file)
            else:
                classes = torch_load(classes_file)
            groups = (classes == 2).logical_or(classes == 3)
            feature_idx = groups.nonzero().view(-1)
            nonfeature_idx = groups.logical_not().nonzero().view(-1)
            feature_idx_n = feature_idx.size(0)
            half_count = min(groups.size(0) - feature_idx_n, feature_idx_n)
            indices = torch_cat([
                feature_idx[torch_randperm(feature_idx_n)[:half_count]],
                nonfeature_idx[torch_randperm(nonfeature_idx.size(0))[:half_count]]
            ])
            for level in _nl.gaussian:
                file_location = _pdir / (_stem + self.GAUSSIAN + str(level) + self.EXTENSION)
                if not file_location.exists():
                    noise.generateNoise(level, 0, 0)
                    data_list = preprocessor.getGraphs(indices)
                    store = tg_data_Batch.from_data_list(data_list)
                    torch_save(store, str(file_location))
            for level in _nl.impulsive:
                file_location = _pdir / (_stem + self.IMPULSIVE + str(level) + self.EXTENSION)
                if not file_location.exists():
                    noise.generateNoise(level, 1, 0)
                    data_list = preprocessor.getGraphs(indices)
                    store = tg_data_Batch.from_data_list(data_list)
                    torch_save(store, str(file_location))

# Example usage:
# dataset = PatchDataset(root='./data')

# Apply transformations to the data if desired
# if dataset.transform is not None:
#     dataset.transform = T.Compose([
#         T.RandomRotate(30, resample=False),
#         T.RandomTranslate(0.1)
#     ])

# Put the dataset in a data loader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Loop over the dataloader
# for batch in dataloader:
#     # Access the data for each graph in the batch
#     x, edge_index = batch.x, batch.edge_index
#     # Do something with the data (e.g., train your GNN)
#     print(x, edge_index)
