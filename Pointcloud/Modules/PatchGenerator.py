from .Object import Mesh, Pointcloud, HelperFunctions

from numpy import arange as np_arange, argsort as np_argsort, array as np_array, average as np_average, cross as np_cross, exp as np_exp, max as np_max, sum as np_sum, transpose as np_transpose
from numpy.linalg import det as np_linalg_det, eigh as np_linalg_eigh, norm as np_linalg_norm
from sklearn.preprocessing import normalize as sklearn_preprocessing_normalize
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

    def getRadius(self, tworing):
        _object = self.object
        _g = _object.g
        _edge_index = _g.edge_index
        k = 8
        if isinstance(_object, Mesh):
            a = np_average(_object.fa[tworing])
            radius = k*a**0.5
        else:
            m = torch_logical_and(torch_isin(_edge_index[0], tworing), torch_isin(_edge_index[1], tworing))
            n = np_average(np_linalg_norm((_g.pos[_edge_index[1][m]] - _g.pos[_edge_index[0][m]]).numpy(), axis=1))
            radius = k*n
        return radius

    def getNodes(self, neighbours):
        _object = self.object
        if isinstance(_object, Mesh):
            _vta = _object.vta
            _vta1 = _vta[1]
            # No sort or unique needed. Ranges are unique and sorted.
            ts_i = HelperFunctions.rangeBoundariesToIndices(_vta1[neighbours], _vta1[neighbours+1])
            # Values in vta0 are not unique, because multiple vertices can have the same triangle.
            ts = np_array(list(set(_vta[0][ts_i])))
            return ts
        else:
            return neighbours
    
    # Collecting patches from the object.
    # mode is 0 or 1.
    #   - 0 represents the method from the original paper and;
    #   - 1 represents a new sped up version.
    def toPatchIndices(self, mode=0):
        _object = self.object
        _g = _object.g
        _pos = _g.pos
        if mode == 0:
            print("Started collecting patches: Collecting Tworings")
            tworings = [self.getTwoRing(i) for i in range(_pos.size(0))]
            print("Collected Tworings: Collecting radii")
            # Calculate ball radii
            radii = [self.getRadius(tr) for tr in tworings]
            print("Collected radii: Collecting nearby vertices")
            # Select points in ball
            nearby_vertices = np_array(_object.kdtree.query_ball_point(_pos, radii))
            print("Collected nearby vertices: Collecting nodes from vertices.")
            # Get graph nodes from vertices in range
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in nearby_vertices]
            self.nodes = nodes
            self.mode = mode
            return nodes
        elif mode == 1:
            print("Collecting nearby vertices")
            _, nearby_vertices = _object.kdtree.query(_pos, k=64, workers=-1)
            print("Collecting nodes")
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in nearby_vertices]
            self.nodes = nodes
            self.mode = mode
            return nodes
        
    def alignPatchIndices(self, indices):
        # This sigma should be changed in the future maybe! This is the best guess for what sigma should be currently..
        # Proof has been found in the code that sigma should indeed be a third!
        # Since sigma will only be used at one place, where it is used as a denominator, the inverse of sigma is stored.
        SIGMA_1 = 3
        _object = self.object
        # (N, 3)
        bcs = _object.g.pos.numpy()
        # (N, 3)
        ci = bcs
        # (N, 64, 3)
        cj = bcs[indices]
        # (N, 64, 3)
        cjci = cj - ci[:, None, :]
        # (N,)
        scale_factors = 1 / np_max(np_linalg_norm(cjci, axis=2), axis=1)
        # (N, 64, 3) (Translated and scaled)
        dcs = cjci * scale_factors[:, None, None]
        # (N, 3)
        n = _object.fn if isinstance(_object, Mesh) else _object.vn
        # (N, 64, 3)
        nj = n[indices]
        # (N, 64, 3)
        wj = np_cross(np_cross(dcs, nj, axis=2), dcs).reshape(-1, 3) # Reshape is done for normalize method to work
        sklearn_preprocessing_normalize(wj, copy=False) # Normalize wj in place
        wj = wj.reshape(-1, 64, 3)
        # (N, 64, 3)
        njprime = 2 * np_sum(nj * wj, axis=2)[:, :, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, is doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (N, 64)
        areas = (_object.fa if isinstance(_object, Mesh) else _object.va)[indices] * scale_factors[:, None] ** 2
        # (N,)
        maxArea = np_max(areas, axis=1)
        # (N, 64)
        ddcs = np_linalg_norm(dcs, axis=2)
        # (N, 64)
        muj = (areas / maxArea[:, None])*np_exp(-ddcs*SIGMA_1)
        # (N, 64, 3, 3)
        outer = njprime[..., None] * njprime[..., None, :]
        # (N, 64, 3, 3)
        Tj = muj[..., None, None] * outer
        # (N, 3, 3)
        Ti = np_sum(Tj, axis=1)
        # ((N, 3), (N, 3, 3))
        eigh = np_linalg_eigh(Ti)
        # (N, 3)
        ev_order = np_argsort(eigh[0], axis=1)[:, ::-1]
        # (N, 3, 3)
        eigh_T = np_transpose(eigh[1], axes=(0, 2, 1))
        # (N, 3, 3)
        matrix = eigh_T[np_arange(ev_order.shape[0])[:, None, None], ev_order[..., None], np_arange(3)[None, None]]
        # (N,)
        mask = np_sum(matrix[:, 0, :] * n, axis=1) < 0
        matrix[mask, 0] *= -1
        # (N,)
        mask_2 = np_linalg_det(matrix) < 0
        matrix[mask_2, 2] *= -1
        '''
            Use matrix to rotate and to find characteristics of patch.
        '''
        # Get nodes indices to select
        # Calculate Normal Voting Tensors
        # Rotate all subgraphs with respective Normal Voting Tensors
        # IN WHAT ORDER DO I DO THINGS WTFFFFFFFFFFFFFFFFFF
