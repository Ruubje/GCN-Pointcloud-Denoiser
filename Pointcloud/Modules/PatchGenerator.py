from .Object import Pointcloud, HelperFunctions

from numpy import arange as np_arange, array as np_array, average as np_average
from numpy.linalg import norm as np_linalg_norm
from torch import argwhere as torch_argwhere, cat as torch_cat, isin as torch_isin, logical_and as torch_logical_and, unique as torch_unique
from torch_geometric.utils import subgraph as torch_geometric_utils_subgraph

class PatchGenerator:
    
    def __init__(self, object):
        if not isinstance(object, Pointcloud):
            raise ValueError(f"A Patch Generator expects a Pytorch Geometric Data object from which it can create patches.")

        self.object = object

    def getTwoRing(self, i):
        _edge_index = self.object.g.edge_index
        j = torch_isin(_edge_index[0], i)
        neighbors = _edge_index[1][j].view(-1)
        l = torch_isin(_edge_index[0], neighbors)
        neighbors2 = _edge_index[1][l].view(-1)
        return torch_unique(torch_cat((neighbors, neighbors2)))


    def getRadius(self, tworing, mode=0):
        _object = self.object
        _g = _object.g
        _edge_index = _g.edge_index
        k = 8
        if mode == 0:
            m = torch_logical_and(torch_isin(_edge_index[0], tworing), torch_isin(_edge_index[1], tworing))
            n = np_average(np_linalg_norm((_g.pos[_edge_index[1][m]] - _g.pos[_edge_index[0][m]]).numpy(), axis=1))
            radius = k*n
        elif mode == 1:
            a = np_average(_object.a[tworing])
            radius = k*a**0.5
        return radius

    def getNodes(self, neighbours, mode=0):
        _object = self.object
        if mode == 0:
            return neighbours
        elif mode == 1:
            _vta = _object.vta
            _vta0 = _vta[0]
            _vta1 = _vta[1]
            # No sort or unique needed. Ranges are unique and sorted.
            ts_i = HelperFunctions.rangeBoundariesToIndices(_vta1[neighbours], _vta1[neighbours+1])
            # Values in vta0 are not unique, because multiple vertices can have the same triangle.
            ts = np_array(list(set(_vta0[ts_i])))
            return ts
        
    def toPatches(self):
        _object = self.object
        _g = _object.g
        _pos = _g.pos
        print("Started collecting patches: Collecting Tworings")
        tworings = [self.getTwoRing(i) for i in range(_pos.size(0))]
        print("Collected Tworings. Starting to collect radii")
        # Calculate ball radii
        radii = [self.getRadius(tr, 0) for tr in tworings]
        print("Collected radii, starting the ball query")
        # Select points in ball
        nearby_vertices = np_array(_object.kdtree.query_ball_point(_pos, radii))
        print("Collected bearby vertices. Starting to collect nodes from vertices.")
        # Get graph nodes from vertices in range
        nodes = [self.getNodes(np_array(neighbours), 0) for neighbours in nearby_vertices]
        return nodes
        # Get nodes indices to select
        # Calculate Normal Voting Tensors
        # Rotate all subgraphs with respective Normal Voting Tensors
        # IN WHAT ORDER DO I DO THINGS WTFFFFFFFFFFFFFFFFFF
