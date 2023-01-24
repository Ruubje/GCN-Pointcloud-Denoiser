import igl
import numpy as np
from Modules.Utils import Mesh
import meshplot as mp

import timeit

# PatchCollector is a class that creates and keeps track of patches from a mesh.
class PatchCollector:
    
    # Collectors are initialized with the mesh they are working on.
    # mesh is a Mesh object that represents the mesh model that we are working on.
    # k is an optional parameter that increases that patch size manually.
    def __init__(self, mesh, k=4):
        self.k = k
        self.mesh = mesh
    
    # This method returns a Mesh object that contains the patch corresponding to the face.
    # fi is the face index from which we want to create a patch.
    # Returns a new Mesh object containing the selected patch.
    def selectPaperPatch(self, fi):
        nh = self.mesh.getNeighbourhood(fi, 2)
        avg_tworing_area = np.sum(self.mesh.getAreas(nh)) / nh.shape[0]
        r = self.k * avg_tworing_area ** 0.5
        faces = self.mesh.getFacesInRange(np.sum(self.mesh.v[self.mesh.f[fi]], axis=0) / 3, r)
        patch, imv = self.mesh.select_faces(faces)
        old_vertices_of_face = self.mesh.f[fi]
        new_vertices_of_face = imv[old_vertices_of_face]
        mask_where_new_vertices_are_equal_to_new_vertices_of_face = np.equal(patch.f, new_vertices_of_face)
        mask_where_all_vertices_are_equal = np.all(mask_where_new_vertices_are_equal_to_new_vertices_of_face, axis=1)
        faces_where_all_vertices_are_equal = np.arange(len(patch.f))[mask_where_all_vertices_are_equal]
        patch.pi = faces_where_all_vertices_are_equal
        return patch

    # Transforms / Aligns the patch as described in the paper.
    # mesh is a Mesh object representing a patch that needs to be aligned.
    def alignPatch(self, mesh):
        mesh.translate(-1*mesh.getPCCenter())
        mesh.resize(mesh.getPCSize() / np.max(mesh.getPCBoundingBox()))
        rotationMatrix = self.getPaperRotationMatrix(mesh)
        mesh.rotate(rotationMatrix)

    # Get the rotation matrix for a given patch. The algorithm for defining the matrix is described in the paper.
    # patch is a Mesh object representing a patch from which a rotation matrix needs to be retrieved.
    # Returns a 3x3 array containing the rotation matrix.
    def getPaperRotationMatrix(self, patch):
        # Can only execute if attribute pi is set with a face id.
        bcs = igl.barycenter(patch.v, patch.f)
        ci = bcs[patch.pi]
        cj = np.delete(bcs, patch.pi, axis=0)
        dcs = cj - ci
        nj = np.delete(patch.getFaceNormals(), patch.pi, axis=0)
        raw_wj = np.cross(np.cross(dcs, nj, axis=1), dcs)
        wj = np.nan_to_num(raw_wj / np.linalg.norm(raw_wj, axis=1)[:, None])
        njprime = 2 * np.sum(np.multiply(nj, wj), axis=1)[:, None] * wj - nj
        areas = patch.getAreas(np.delete(np.arange(len(patch.f)), patch.pi, axis=0))
        maxArea = np.max(areas)
        ddcs = np.linalg.norm(dcs, axis=1)
        # This sigma should be changed in the future maybe! This is the best guess for what sigma should be currently..
        sigma = 1./3.
        muj = (areas / maxArea)*np.exp(-ddcs/sigma)
        outer = njprime[..., None] * njprime[:, None]
        Tj = muj[:, None, None] * outer
        Ti = np.sum(Tj, axis=0)
        eig = np.linalg.eig(Ti)
        sort = np.flip(np.argsort(eig[0]))
        return eig[1][sort]

    def collectAllPatches(self):
        numberOfFaces = len(self.mesh.f)
        print("Start selecting patches")
        selectedPatches = []
        for i in range(numberOfFaces):
            selectedPatches.append(self.selectPaperPatch(i))
            print("Patch " + str(i) + "/" + str(numberOfFaces) + " selected!")
        for i, patch in enumerate(selectedPatches):
            self.alignPatch(patch)
            print("Patch " + str(i) + "/" + str(numberOfFaces) + " aligned!")
        return selectedPatches
