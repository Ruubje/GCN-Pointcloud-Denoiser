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
    newMesh = myMesh.copy()
    assert not (newMesh is myMesh)
    for attribute, value1 in vars(myMesh).items():
        value2 = vars(newMesh)[attribute]
        assert (value1 is None) or (value2 is None) or not (value1 is value2)
    
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
