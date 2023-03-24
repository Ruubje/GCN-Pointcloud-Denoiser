import igl
import numpy as np
from Modules.Mesh import Mesh
import meshplot as mp

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
        faces_in_range = self.mesh.getFacesInRange(np.sum(self.mesh.v[self.mesh.f[fi]], axis=0) / 3, r)
        patch = self.mesh.createPatch(faces_in_range, fi)
        return patch

    def collectAllPatches(self):
        numberOfFaces = len(self.mesh.f)
        print("Start selecting patches")
        selectedPatches = []
        for i in range(numberOfFaces):
            selectedPatches.append(self.selectPaperPatch(i))
            print("Patch " + str(i) + "/" + str(numberOfFaces) + " selected!")
        for i, patch in enumerate(selectedPatches):
            patch.alignPatch()
            print("Patch " + str(i) + "/" + str(numberOfFaces) + " aligned!")
        return selectedPatches
