import numpy as np
from .Mesh import Mesh
import pathlib
import time
from .Network.DataUtils import PatchDataset

# PatchCollector is a class that creates and keeps track of patches from a mesh.
class PatchCollector:
    
    # Collectors are initialized with the mesh they are working on.
    # mesh is a Mesh object that represents the mesh model that we are working on.
    # k is an optional parameter that increases that patch size manually.
    def __init__(self, mesh, k=4):
        self.k = k
        self.mesh = mesh
    
    # Reading an obj file directly into a PatchCollector object.
    # obj_file is a String representing the path towards an Object (.obj) file.
    # Returns a new PatchCollector object
    @classmethod
    def readFile(cls, obj_file, k=4):
        mesh = Mesh.readFile(obj_file)
        return PatchCollector(mesh, k)

    # This method returns a Mesh object that contains the patch corresponding to the face.
    # fi is the face index from which we want to create a patch.
    # Returns a new Mesh object containing the selected patch.
    def selectPaperPatch(self, fi):
        nh = self.mesh.getNeighbourhood(fi, 2)
        avg_tworing_area = np.sum(self.mesh.getAreas(nh)) / nh.shape[0]
        r = self.k * avg_tworing_area ** 0.5
        center = np.average(self.mesh.getVertices()[self.mesh.f[fi]], axis=0)
        faces_in_range = self.mesh.getFacesInRange(center, r)
        patch = self.mesh.createPatch(faces_in_range, fi)
        return patch

    # Collecting all patches from the mesh. First selecting and storing all patches and then aligning them
    # timeout is a time in seconds. If the timeout time is reached the method stops collecting and starts aligning the collected patches.
    # Returns the aligned patches (array of Patch objects) (within the time limit).
    def collectAllPatches(self, timeout=-1):
        start_time = time.time()
        numberOfFaces = len(self.mesh.f)
        print("Start selecting patches")
        selectedPatches = []
        for i in range(numberOfFaces):
            selectedPatches.append(self.selectPaperPatch(i))
            msg = "Patch " + str(i+1) + "/" + str(numberOfFaces) + " selected!"
            time_since_start = int(time.time() - start_time)
            if timeout > -1:
                msg = f"[Timeout: {time_since_start}/{timeout}] " + msg
            print(msg)
            if time_since_start >= timeout and timeout > -1:
                break
        numberOfSelectedPatches = len(selectedPatches)
        for i, patch in enumerate(selectedPatches):
            patch.alignPatch()
            msg = "Patch " + str(i+1) + "/" + str(numberOfSelectedPatches) + " aligned!"
            print(msg)
        return selectedPatches
    
    # Collects all patches and also saves them to a file directory.
    # directory_path is a string representing the path to the directory where the patches need to be stored.
    #   if the directory doesn't exists, it is created.
    # timeout is passed on to collectAllPatches.
    # Returns nothing.
    def collectAndSavePatches(self, directory_path, timeout=-1):
        pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
        patches = self.collectAllPatches(timeout)
        numberOfPatches = len(patches)
        for index, patch in enumerate(patches):
            level = int(patch.noise_factor*1000)
            file_name = f"{level}_{index}.mat"
            file_path = directory_path + "/" + file_name
            patch.save(file_path)
            msg = "Patch " + str(index+1) + "/" + str(numberOfPatches) + " saved!"
            print(msg)

    # Collects all patches and transforms them into input for a DGCNN.
    def collectNetworkInput(self, timeout=-1):
        patches = self.collectAllPatches(timeout)
        fileformat = np.array([PatchDataset.file2input(*patch.toGraph(), 64) for patch in patches])
        print(fileformat.shape)