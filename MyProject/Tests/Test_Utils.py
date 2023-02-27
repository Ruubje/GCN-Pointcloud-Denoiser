import pytest
from Modules.Utils import *
import numpy as np
import igl

@pytest.fixture
def myMesh():
    v = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0]
    ], dtype=np.float32)
    f = np.array([
        [1, 0, 2],
        [1, 2, 3],
        [3, 2, 4],
        [3, 4, 5],
        [4, 0, 5],
        [0, 1, 5],
        [2, 0, 4],
        [1, 3, 5]
    ])
    return Mesh(v, f)

def test_setup_example_mesh(myMesh):
    assert myMesh.v.shape == (6, 3)
    assert myMesh.f.shape == (8, 3)

def test_normals_example_mesh(myMesh):
    normals = myMesh.getFaceNormals()
    barycenters = igl.barycenter(myMesh.v, myMesh.f)
    dot_product = np.sum(normals * barycenters, axis=1)
    print(normals.shape, barycenters.shape, dot_product.shape)
    assert np.all(dot_product > 0)

def test_select_faces_remove_5(myMesh):
    newMesh, imv = myMesh.select_faces([0, 1, 2, 6])
    assert newMesh.v.shape[0] == 5
    assert imv[5] == -1

def test_select_faces_remove_3_and_4(myMesh):
    newMesh, imv = myMesh.select_faces([0, 5])
    assert newMesh.v.shape[0] == 4
    assert imv[3] == -1
    assert imv[4] == -1

def test_copy_all_values_are_different_objects_or_values(myMesh):
    # Setup attribute values'
    _ = myMesh.getFaceNormals()
    _ = myMesh.getNeighbourhood(0, 1)
    myMesh.pi = 0
    _ = myMesh.getVertexTriangleAdjacency()
    # Copy and assert
    newMesh = myMesh.copy()
    assert not (newMesh is myMesh)
    for attribute, value1 in vars(myMesh).items():
        value2 = vars(newMesh)[attribute]
        assert not (value1 is None) and not (value2 is None)
        assert not (value1 is value2) or (value1 == value2 and type(value1) == int)
    
def test_getPCCenter(myMesh):
    center = myMesh.getPCCenter()
    assert np.all(center == 0.0)
    
def test_getPCSize(myMesh):
    size = myMesh.getPCSize()
    assert size == 1.0

def test_getPCBoundingBox(myMesh):
    size = myMesh.getPCBoundingBox()
    assert np.all(size == np.array([2, 2, 2]))
    
def test_getAreas(myMesh):
    areas = myMesh.getAreas(np.arange(len(myMesh.f)))
    assert areas.shape[0] == myMesh.f.shape[0]
    assert np.all(areas == np.sqrt(3)/2)

def test_getFacesInRange(myMesh):
    center = np.array([2.0, 0.0, 0.0])
    range = 1.5
    faces_in_range = myMesh.getFacesInRange(center, range)
    assert np.all(faces_in_range == np.array([0, 1, 2, 6]))

def test_getNeighbourhood_wrong_ring(myMesh):
    face_index = 0
    ring = -10
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert neighbourhood.shape[0] == 0
    
def test_getNeighbourhood_1_ring(myMesh):
    face_index = 0
    ring = 1
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.array([0, 1, 5, 6]))

def test_getNeighbourhood_2_ring(myMesh):
    face_index = 0
    ring = 2
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.array([0, 1, 2, 4, 5, 6, 7]))

def test_getNeighbourhood_3_ring(myMesh):
    face_index = 0
    ring = 3
    neighbourhood = myMesh.getNeighbourhood(face_index, ring)
    assert np.all(neighbourhood == np.arange(myMesh.f.shape[0]))

def test_getFaceNormals(myMesh):
    face_normals = myMesh.getFaceNormals()
    sqrt1div3 = np.sqrt(1/3)
    expected_result = np.array([
        [sqrt1div3, sqrt1div3, sqrt1div3],
        [sqrt1div3, sqrt1div3, -sqrt1div3],
        [sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, -sqrt1div3],
        [-sqrt1div3, -sqrt1div3, sqrt1div3],
        [-sqrt1div3, sqrt1div3, sqrt1div3],
        [sqrt1div3, -sqrt1div3, sqrt1div3],
        [-sqrt1div3, sqrt1div3, -sqrt1div3]
    ])
    assert np.all(np.square(face_normals - expected_result) < 0.001)

def test_getFaceNormals_Twice(myMesh):
    face_normals = myMesh.getFaceNormals()
    assert face_normals is myMesh.getFaceNormals()
    assert myMesh.n is face_normals

def test_getVertexTriangleAdjacency(myMesh):
    vta = myMesh.getVertexTriangleAdjacency()
    expected_result_0 = np.array([0, 4, 5, 6, 0, 1, 5, 7, 0, 1, 2, 6, 1, 2, 3, 7, 2, 3, 4, 6, 3, 4, 5, 7], dtype=np.int32)
    expected_result_1 = np.array([0, 4, 8, 12, 16, 20, 24], dtype=np.int32)
    assert np.all(vta[0] == expected_result_0)
    assert np.all(vta[1] == expected_result_1)

def test_getVertexTriangleAdjacency_call_twice(myMesh):
    vta = myMesh.getVertexTriangleAdjacency()
    assert vta is myMesh.getVertexTriangleAdjacency()

def test_getTrianglesOfVertex(myMesh):
    vi = 0
    triangles = myMesh.getTrianglesOfVertex(vi)
    expected_result = np.array([0, 4, 5, 6])
    assert np.all(triangles == expected_result)

def test_getTrianglesOfVertices(myMesh):
    vis = np.array([0, 1])
    triangles = myMesh.getTrianglesOfVertices(vis)
    expected_result = np.array([0, 1, 4, 5, 6, 7])
    assert np.all(triangles == expected_result)

def test_translate_(myMesh):
    translation = np.array([0.5, 2.5, 1.3])
    old_v = np.copy(myMesh.v)
    size = myMesh.getPCSize()

    myMesh.translate(translation)

    assert np.all(np.square(myMesh.v - translation - old_v) < 0.001)
    assert np.square(size - myMesh.getPCSize()) < 0.001

def test_resize(myMesh):
    new_size = 100
    expected_result = np.array([
        [0, 0, 100],
        [0, 100, 0],
        [100, 0, 0],
        [0, 0, -100],
        [0, -100, 0],
        [-100, 0, 0]
    ], dtype=np.float32)

    myMesh.resize(new_size)

    assert np.all(myMesh.v == expected_result)

def test_rotate(myMesh):
    rotation = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    expected_result = np.array([
        [0, -1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0]
    ], dtype=np.float32)

    myMesh.rotate(rotation)

    assert np.all(myMesh.v == expected_result)