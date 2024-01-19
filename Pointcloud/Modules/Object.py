from copy import deepcopy
from igl import barycenter as igl_barycenter,\
    doublearea as igl_doublearea,\
    read_obj as igl_read_obj,\
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency,\
    per_vertex_normals as igl_per_vertex_normals,\
    per_face_normals as igl_per_face_normals,\
    triangle_triangle_adjacency as igl_triangle_triangle_adjacency,\
    vertex_triangle_adjacency as igl_vertex_triangle_adjacency
from numpy import any as np_any,\
    arange as np_arange,\
    argwhere as np_argwhere,\
    array as np_array,\
    c_ as np_c_,\
    cross as np_cross,\
    logical_not as np_logical_not,\
    nan_to_num as np_nan_to_num,\
    ones as np_ones,\
    repeat as np_repeat,\
    sort as np_sort,\
    sum as np_sum,\
    vstack as np_vstack,\
    unique as np_unique,\
    zeros as np_zeros
from pathlib import Path
#https://github.com/nmwsharp/robust-laplacians-py
from robust_laplacian import point_cloud_laplacian
from scipy.spatial import KDTree as scipy_spatial_KDTree
from sklearn.preprocessing import normalize as sklearn_preprocessing_normalize
from torch import arange as torch_arange,\
    from_numpy as torch_from_numpy,\
    long as torch_long,\
    stack as torch_stack
from torch_geometric.data import Data as pyg_data_Data
from warnings import warn

'''
    This file contains classes that represent .obj files.
    A mesh is a pointcloud connected with triangles and therefore
    it extends the pointcloud (and sometimes overrides functions)
'''

class Object:

    '''
        Init stuff
    '''

    def __init__(self, file_path, read_file=True):
        if not isinstance(file_path, str):
            raise ValueError(f"file_path: ({file_path}) should be a string.")
        self.file_path = Path(file_path)
        if self.file_path.suffix != ".obj" or not self.file_path.exists() or not self.file_path.is_file():
            raise FileNotFoundError(f"File {file_path} is not a .obj file or is not found.")
        if read_file:
            return self.readFile()
     
    def readFile(self):
        file_path = self.file_path
        if file_path.is_file():
            data = igl_read_obj(str(file_path))
            self.gt = data[0]
            return data

class Pointcloud(Object):

    DEFAULT_NEIGHBOURHOOD_MODE = 1

    '''
        Init Stuff
    '''

    def __init__(self, file_path, read_file=True):
        super().__init__(file_path, read_file=False)
        self.graph_vertices_match = False
        if read_file:
            self.readFile()

    def readFile(self, mode=DEFAULT_NEIGHBOURHOOD_MODE, calculate_meta=True):
        data = super().readFile()
        _data0 = data[0]
        _data3 = data[3]
        self.f = _data3
        self.vta = igl_vertex_triangle_adjacency(_data3, _data0.shape[0])
        self.setVertices(_data0)
        if calculate_meta:
            self.setGraph(mode=mode)
            self.g.y = torch_from_numpy(deepcopy(self.vn))
        return data
    
    '''
        Object stuff
    '''
    
    def setVertices(self, v, mode=DEFAULT_NEIGHBOURHOOD_MODE, calculate_meta=True):
        self.graph_vertices_match = False
        self.v = v
        self.kdtree = scipy_spatial_KDTree(v)
        self.calculateNormals()
        self.calculateAreas()
        if calculate_meta:
            self.setGraph(mode=mode)

    def calculateNormals(self):
        self.vn = np_nan_to_num(igl_per_vertex_normals(self.v, self.f), copy=False, nan=0)
        self.fn = igl_per_face_normals(self.v, self.f, Mesh.DEGENERATE_NORMAL_PLACEHOLDER)

    def calculateAreas(self):
        if self.vta is None:
            raise AttributeError("Attribure 'vta' not found.")
        _vta = self.vta
        _vta1 = _vta[1]
        self.fa = igl_doublearea(self.v, self.f) / 2.0
        self.va = np_array([np_sum(self.fa[_vta[0][_vta1[vi]:_vta1[vi+1]]]) / 3 for vi in np_arange(len(self.v))])
    
    def getNormals(self):
        return self.vn

    def getAreas(self):
        return self.va
    
    '''
        Graph stuff
    '''

    def setGraph(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        old_y = None
        if hasattr(self, 'g') and hasattr(self.g, 'y'):
            old_y = self.g.y
        self.g = self.toGraph(mode=mode)
        self.g.y = old_y
        self.graph_vertices_match = True

    def toNodes(self):
        return torch_from_numpy(self.v)
    
    # Create a graph from the pointcloud
    # mode: 0 is knn and 1 is 
    def toEdges(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        if mode == 0:
            KNN_K_HARDCODED = 12
            # k=12, because the mean + var degree of the armadillo & fandisk are 8 & 10 respectively.
            # This means that knn for k=12 will cover most of the details that the robust laplacian will catch as well!
            return HelperFunctions.toEdgeTensor(self.kdtree.query(self.v, k=KNN_K_HARDCODED+1)[1][:, 1:])
        elif mode == 1:
            L, _ = point_cloud_laplacian(self.v)
            Lcoo = L.tocoo()
            return torch_from_numpy(np_vstack((Lcoo.row, Lcoo.col)))
        else:
            raise ValueError(f"mode {mode} is undefined within this method.")
    
    def toGraph(self, mode=DEFAULT_NEIGHBOURHOOD_MODE):
        return pyg_data_Data(edge_index=self.toEdges(mode=mode), pos=self.toNodes())

class Mesh(Pointcloud):

    DEGENERATE_NORMAL_PLACEHOLDER = np_zeros(3)

    '''
        Init Stuff
    '''

    def __init__(self, file_path, read_file=True):
        super().__init__(file_path, read_file=False)
        if read_file:
            self.readFile()

    def readFile(self, calculate_meta=True):
        data = super().readFile(calculate_meta=False)
        if calculate_meta:
            self.calculateNormals()
            self.setGraph()
            self.g.y = torch_from_numpy(deepcopy(self.fn))
        return data
    
    '''
        Object stuff
    '''

    def setVertices(self, v, calculate_meta=True):
        super().setVertices(v, calculate_meta=False)
        if calculate_meta:
            self.toGraph()

    def getNormals(self):
        return self.fn
    
    def getAreas(self):
        return self.fa

    '''
        Graph stuff
    '''

    def setGraph(self):
        old_y = None
        if hasattr(self, 'g') and hasattr(self.g, 'y'):
            old_y = self.g.y
        self.g = self.toGraph()
        self.g.y = old_y
        self.graph_vertices_match = True

    def toNodes(self):
        return torch_from_numpy(igl_barycenter(self.v, self.f))
    
    def toEdges(self):
        num_f = self.f.shape[0]
        tta = igl_triangle_triangle_adjacency(self.f)[0]
        return torch_from_numpy(np_c_[np_repeat(np_arange(num_f)[:, None], 3, axis=1).flatten(), tta.flatten()].T)
    
    def toGraph(self):
        return pyg_data_Data(edge_index=self.toEdges(), pos=self.toNodes())

class HelperFunctions:

    DEGENERATE_NORMAL_PLACEHOLDER = np_zeros(3)

    @classmethod
    def calculateVertexNormals(cls, v, f):
        return np_nan_to_num(igl_per_vertex_normals(v, f), copy=False, nan=0)

    @classmethod
    def calculateFaceNormals(cls, v, f):
        return igl_per_face_normals(v, f, cls.DEGENERATE_NORMAL_PLACEHOLDER)
    
    @classmethod
    # https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
    # starts: 1D numpy array of indices of size n representing the starts of the ranges.
    # ends: 1D numpy array of indices of size n representing the ends of the ranges.
    # return: 1D numpy array of indices with indices that are within one of the given ranges.
    # WARNING: The list can be unsorted and contain duplicates.
    def rangeBoundariesToIndices(cls, starts, ends):
        l = ends - starts
        nonsense_ids = l <= 0
        if np_any(nonsense_ids):
            # Lengths of ranges that are zero are nonsense and should be ignored.
            # Ranges with a negative length have the start and end reversed.
            #   This should be fixed before calling this function for efficiency.
            warn(f"Nonsensible ranges are given and will be ignored! (IDs: {np_arange(len(nonsense_ids))[nonsense_ids]}, Range lengths: {l[nonsense_ids]})")
            sensible_ids = np_logical_not(nonsense_ids)
            starts = starts[sensible_ids]
            ends = ends[sensible_ids]
            l = ends - starts
        clens = l.cumsum()
        ids = np_ones(clens[-1],dtype=int)
        ids[0] = starts[0]
        ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
        return ids.cumsum()
    
    @classmethod
    def toEdgeTensor(cls, output):
        col = torch_from_numpy(output).to(torch_long)
        k = col.size(1)
        row = torch_arange(col.size(0), dtype=torch_long).view(-1, 1).repeat(1, k)
        return torch_stack([row.reshape(-1), col.reshape(-1)], dim=0)
    
    '''
        DEPRECATED DEFINITIONS
    '''

    # v: (num_vertices, num_axes) -> Euclidian position
    # f: (num_faces, num_corners) -> vertex_index
    # n: (num_faces, num_axes) -> Euclidian position
    @classmethod
    def vertexNormalsFromFaceNormals(cls, v, f, n):
        warn("This method is slow and therefore deprecated! Use igl.per_vertex_normals instead!")
        # vta: ((3*num_faces,), (num_vertices+1)) -> (neighbor_face_index, cumulative_vertex_degree)
        vta = igl_vertex_triangle_adjacency(f, len(v))
        # vi: (num_vertices,) -> all vertex indices
        vi = np_arange(len(v))
        # v_degree: (num_vertices,) -> vertex_degree
        v_degree = vta[1][vi+1] - vta[1][vi]
        # unique_degrees: (num_unique_vertex_degrees,) -> vertex_degree
        # vi_di: (num_vertices,) -> unique_vertex_degree_index
        unique_degrees, vi_di = np_unique(v_degree, return_inverse=True)
        vertex_normals = np_zeros((len(v), 3))
        # di: degree_i
        # degree: vertex_degree
        for di, unique_degree in enumerate(unique_degrees):
            # v_with_degree: (num_vertices_with_degree,) -> vertex_index
            v_with_degree = np_argwhere(vi_di == di).reshape(-1)
            # vta_to_index: (num_faces_of_vertices_with_degree,) -> vta_index
            vta_to_index = cls.rangeBoundariesToIndices(vta[1][v_with_degree].reshape(-1), vta[1][v_with_degree+1].reshape(-1))
            # faces_of_v_with_degree: (num_vertices_with_degree, unique_degree) -> face_index
            faces_of_v_with_degree = vta[0][vta_to_index].reshape(-1, unique_degree)
            # normals_of_faces: (num_vertices_with_degree, unique_degree, num_axes) -> Euclidian position
            normals_of_faces = n[faces_of_v_with_degree]
            # normalized_summed_normals: (num_vertices_with_degree, num_axes) -> Euclidian position
            normalized_summed_normals = sklearn_preprocessing_normalize(normals_of_faces.sum(axis=1))
            vertex_normals[v_with_degree] = normalized_summed_normals
        return vertex_normals
    
    # v: (num_vertices, num_axes) -> Euclidian position
    # f: (num_faces, num_corners) -> vertex_index
    @classmethod
    def normalsOfFaces(cls, f, v):
        warn("This method is slow and therefore deprecated! Use igl.per_face_normals instead!")
        # fv: (num_faces, num_corners, num_axes) -> Euclidian position
        fv = v[f]
        # crosses: (num_faces, num_axes) -> Euclidian position
        normals = sklearn_preprocessing_normalize(np_cross(fv[:, 1, :] - fv[:, 0, :], fv[:, 2, :] - fv[:, 1, :]))
        return normals
