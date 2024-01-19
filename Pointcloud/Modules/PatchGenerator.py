from .Object import \
    Mesh,\
    Pointcloud,\
    HelperFunctions

from numpy import \
    arange as np_arange,\
    argmax as np_argmax,\
    argsort as np_argsort,\
    array as np_array,\
    average as np_average,\
    cross as np_cross,\
    einsum as np_einsum,\
    exp as np_exp,\
    logical_and as np_logical_and,\
    max as np_max,\
    ndarray as np_ndarray,\
    sort as np_sort,\
    sum as np_sum,\
    transpose as np_transpose,\
    zeros as np_zeros
from numpy.linalg import \
    det as np_linalg_det,\
    eigh as np_linalg_eigh,\
    norm as np_linalg_norm
from sklearn.preprocessing import \
    normalize as sklearn_preprocessing_normalize
from torch import \
    arange as torch_arange,\
    bool as torch_bool,\
    cat as torch_cat,\
    from_numpy as torch_from_numpy,\
    int64 as torch_int64,\
    isin as torch_isin,\
    logical_and as torch_logical_and,\
    unique as torch_unique,\
    zeros as torch_zeros
from torch_geometric.data import \
    Data as pyg_data_Data
from torch_geometric.utils import \
    degree as pyg_utils_degree,\
    subgraph as pyg_utils_subgraph
from itertools import \
    zip_longest as itertools_zip_longest

class PatchGenerator:

    DEFAULT_PATCH_RESOLUTION_K = 8
    
    def __init__(self, object):
        if not isinstance(object, Pointcloud):
            raise ValueError(f"A Patch Generator expects a Pytorch Geometric Data object from which it can create patches.")

        self.object = object

    '''
        Patch Selection
    '''

    def getTwoRing(self, i):
        _edge_index = self.object.g.edge_index
        j = torch_isin(_edge_index[0], i)
        neighbors = _edge_index[1][j].view(-1)
        l = torch_isin(_edge_index[0], neighbors)
        neighbors2 = _edge_index[1][l].view(-1)
        return torch_unique(torch_cat((neighbors, neighbors2)))

    def getRadius(self, tworing, k=DEFAULT_PATCH_RESOLUTION_K):
        _object = self.object
        a = np_average(_object.getAreas()[tworing])
        radius = k*a**0.5
            # _g = _object.g
            # _edge_index = _g.edge_index
            # Code that calculates the average distance to neighbours and calculates a radius from that metric.
            # m = torch_logical_and(torch_isin(_edge_index[0], tworing), torch_isin(_edge_index[1], tworing))
            # n = np_average(np_linalg_norm((_g.pos[_edge_index[1][m]] - _g.pos[_edge_index[0][m]]).numpy(), axis=1))
            # radius = k*n
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
        
    def toMasked2DArray(self, indices):
        # https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
        indices_2d = np_array(list(itertools_zip_longest(*indices, fillvalue=-1))).T
        mask = indices_2d == -1 # Detect mask
        indices_2d[mask] = 0 # Set mask to center node to not throw errors
        return indices_2d, mask
    
    # Collecting patches from the object.
    # mode is 0 or 1.
    #   - 0 represents the method from the original paper and;
    #   - 1 represents a new sped up version.
    def toPatchIndices(self, mode=0, k=DEFAULT_PATCH_RESOLUTION_K):
        _object = self.object
        _g = _object.g
        _pos = _g.pos
        if mode == 0:
            print("Started collecting patches: Collecting Tworings")
            tworings = [self.getTwoRing(i) for i in range(_pos.size(0))]
            print("Collected Tworings: Collecting radii")
            # Calculate ball radii
            radii = [self.getRadius(tr, k) for tr in tworings]
            print("Collected radii: Collecting nearby vertices")
            # Select points in ball
            nearby_vertices = np_array(_object.kdtree.query_ball_point(_pos, radii))
            print("Collected nearby vertices: Collecting nodes from vertices.")
            # Get graph nodes from vertices in range
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in nearby_vertices]
            nodes, mask = self.toMasked2DArray(nodes)
            return nodes, mask
        elif mode == 1:
            print("Collecting nearby vertices")
            _, nearby_vertices = _object.kdtree.query(_pos, k=64, workers=-1)
            print("Collecting nodes")
            nodes = [self.getNodes(np_array(neighbours)) for neighbours in nearby_vertices]
            nodes, mask = self.toMasked2DArray(nodes)
            return nodes, mask
    
    '''
        Patch Alignment
    '''

    def getEigh(self, indices_2d, mask):
        SIGMA_1 = 3
        _object = self.object

        P = indices_2d.shape[1]
        # (N, 3)
        ci = _object.g.pos.numpy()
        # (N, P, 3)
        cj = ci[indices_2d]
        # (N, P, 3)
        cjci = cj - ci[:, None, :]
        temp_norms = np_linalg_norm(cjci, axis=2)
        temp_norms[mask] = 0
        # (N,)
        scale_factors = 1 / np_max(temp_norms, axis=1)
        # (N, P, 3) (Translated and scaled)
        dcs = cjci * scale_factors[:, None, None]
        # (N, 3)
        n = _object.getNormals()
        # (N, P, 3)
        nj = n[indices_2d]
        # (N, P, 3)
        wj = np_cross(np_cross(dcs, nj, axis=2), dcs).reshape(-1, 3) # Reshape is done for normalize method to work
        sklearn_preprocessing_normalize(wj, copy=False) # Normalize wj in place
        wj = wj.reshape(-1, P, 3)
        # (N, P, 3)
        njprime = 2 * np_sum(nj * wj, axis=2)[:, :, None] * wj - nj
        # Big problem. Pointclouds don't have areas. What do we do about this?!?!
        # Reasoning: If the area of the normal vector is small, is doesn't influence the final voting tensor.
        # Should we use neighbourhood density to have a scaling factor or not?
        # (N, P)
        areas = _object.getAreas()[indices_2d] * scale_factors[:, None] ** 2
        print(areas.shape)
        areas[mask] = 0
        # (N,)
        maxArea = np_max(areas, axis=1)
        # (N, P)
        ddcs = np_linalg_norm(dcs, axis=2)
        # (N, P)
        muj = (areas / maxArea[:, None])*np_exp(-ddcs*SIGMA_1)
        # (N, P, 3, 3)
        outer = njprime[..., None] * njprime[..., None, :]
        # (N, P, 3, 3)
        Tj = muj[..., None, None] * outer
        # Before summing, set the nonsense values to zero!
        Tj[mask] = 0
        # (N, 3, 3)
        Ti = np_sum(Tj, axis=1)
        # ((N, 3), (N, 3, 3))
        eigh = np_linalg_eigh(Ti)
        return ci, scale_factors, eigh
    
    @classmethod
    def getGroups(cls, ev):
        N = ev.shape[0]
        ev_f = np_sort(ev, axis=1)[:, ::-1]
        flat = np_logical_and(ev_f[:, 1] < 0.01, ev_f[:, 2] < 0.001)
        edge = np_logical_and(ev_f[:, 1] > 0.01, ev_f[:, 2] < 0.1)
        corner = ev_f[:, 2] > 0.1
        char = np_zeros(N)
        char[flat] = 1
        char[edge] = 2
        char[corner] = 3
        return char
    
    def getRInv(self, eigh):
        N = eigh[0].shape[0]
        _object = self.object
        # (N, 3)
        n = _object.getNormals()
        # (N, 3)
        ev_order = np_argsort(eigh[0], axis=1)[:, ::-1]
        # (N, 3, 3)
        eigh_T = np_transpose(eigh[1], axes=(0, 2, 1))
        # (N, 3, 3)
        R = eigh_T[np_arange(N)[:, None, None], ev_order[..., None], np_arange(3)[None, None]]
        # (N,)
        R[np_sum(R[:, 0, :] * n, axis=1) < 0] *= -1
        # (N,)
        R[np_linalg_det(R) < 0, 2] *= -1
        # (N, 3, 3)
        R_inv = np_transpose(R, axes = (0, 2, 1))
        return R_inv

    @classmethod
    def applyRInv(cls, R_inv, other):
        R_inv_shape = R_inv.shape
        other_shape = other.shape
        len_other_shape = len(other_shape)
        if len(R_inv_shape) == 3 and R_inv_shape[0] == other_shape[0]:
            if len_other_shape == 2:
                return np_einsum(np_einsum("ni,nij->nj", other, R_inv))
            elif len_other_shape == 3:
                return np_einsum("npi,nij->npj", other, R_inv)
        else:
            raise ValueError(f"Input arrays were not of the correct shape:\nR_inv shape: {R_inv_shape}\nOther shape: {other_shape}")
    
    def getGraph(self, index, neighbourhood, alignment):
        p = neighbourhood[index]
        _object = self.object
        _g = _object.g
        nj_R_inv = self.applyRInv(alignment[2], _object.getNormals())
        c = _g.pos[p]
        n = torch_from_numpy(nj_R_inv[i])
        a = torch_from_numpy(_object.getAreas()[p] * S[i])[:, None]
        d = pyg_utils_degree(_g.edge_index[0].to(torch_int64))[p, None]
        x = torch_cat((c, n, a, d), dim=1)
        edge_index = pyg_utils_subgraph(torch_from_numpy(p), _g.edge_index)[0]
        y = gt_R_inv[i]
        data = pyg_data_Data(x=x, edge_index=edge_index, y=y)