import pytest
from Modules.Utils import *
from Modules.PatchCollector import *
import numpy as np
import igl

@pytest.fixture
def myPatch():
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
    return PatchCollector(Mesh(v, f))

def test_selectPaperPatch(myPatch):
    # tba
    return

def test_alignPatch_check_vertices(myPatch):
    patch = myPatch.selectPaperPatch(0)
    myPatch.mesh = patch
    copy = PatchCollector(patch.copy())
    randomRotation = getRandomRotationMatrix()
    copy.mesh.rotate(randomRotation)

    myPatch.alignPatch(myPatch.mesh)
    copy.alignPatch(copy.mesh)

    assert np.allclose(myPatch.mesh.v, copy.mesh.v)
