import igl
import polyscope as ps
import numpy as np
import meshplot as mp

# This class contains methods to manipulate a mesh with vertices and faces.
class Mesh:
    # Initialize the mesh object.
    # v is a list of vertex positions.
    # n is a list of normal vectors.
    # f is a list of indices connecting vertices to create a face.
    # fn is a list of indices connecting vertices to create a face normal???
    # f2f is a map created by igl showing which face is neighbouring which face.
    #   It is only initialized if neighbour look ups are needed.
    # pi wil be set to an index pointing to the face that was selected, which created the current patch.
    def __init__(self, v, f, n = None, f2f = None, pi = None, vta = None):
        self.v = v
        self.f = f
        self.n = n
        self.f2f = f2f
        self.pi = pi
        self.vta = vta

    """
    Class methods
    Contains functionality for when there is no mesh information yet or when there are multiple meshes.
    """

    # Reads an object file and initializes a new Mesh object with it.
    # obj_file is the file path to open.
    # Returns a new Mesh object containing vertex, normal and face information
    @classmethod
    def readFile(cls, obj_file):
        v, _, _, f, _, _ = igl.read_obj(obj_file)
        return Mesh(v, f)

    # View multiple meshes in one visualization as point clouds.
    # *meshes are multiple meshes that need to be shown in one plot.
    # Shows a plot using Meshplot
    @classmethod
    def mpViewPCs(cls, *meshes):
        vertices = None
        color = None
        for i, mesh in enumerate(meshes):
            vertices = mesh.v if i==0 else np.append(vertices, mesh.v, axis=0)
            tiledIndex = np.tile(i, (mesh.v.shape[0], 1))
            color = tiledIndex if i==0 else np.append(color, tiledIndex, axis=0)
        return mp.plot(vertices, c=color, shading={"point_size": np.max(np.linalg.norm(vertices))/1000})
    
    # View multiple meshes in one visualization as meshes.
    # *meshes are multiple meshes that need to be shown in one plot.
    # Shows a plot using Meshplot
    @classmethod
    def mpViewMeshes(cls, *meshes):
        vertices = None
        faces = None
        color = None
        for i, mesh in enumerate(meshes):
            vertices = mesh.v if i==0 else np.append(vertices, mesh.v, axis=0)
            faces = mesh.f if i==0 else np.append(faces, mesh.f + np.tile(vertices.shape[0] - mesh.v.shape[0], mesh.f.shape), axis=0)
            tiledIndex = np.tile(i, (mesh.v.shape[0], 1))
            color = tiledIndex if i==0 else np.append(color, tiledIndex, axis=0)
        return mp.plot(vertices, faces, c=color, shading={"wireframe": True})

    """
    Select faces by mask (array with trues and falses) or id (face ids to keep)
    The methods will return a new mesh object to keep the current data in tact to use for other purposes
    The methods will throw away faces and vertices that are not included in the face selection
    """

    # Creates a new Mesh object containing only information about the selected face indices.
    # faceIds is an array containing face indices to include in the new Mesh.
    # Returns newMesh is the new Mesh object
    #         imv is a map that points the old vertex index to the new vertex index.
    def select_faces(self, faceIds):
        nv, nf, imv, _ = igl.remove_unreferenced(self.v, self.f[faceIds])
        newMesh = Mesh(nv, nf)
        return newMesh, imv

    
    # Creates a copy of the current mesh. Can be used for debugging purposes.
    # Returns a new Mesh object that is a copy of the current object.
    def copy(self):
        v_copy = np.copy(self.v)
        f_copy = np.copy(self.f)
        n_copy = None if self.n is None else np.copy(self.n)
        f2f_copy = None if self.f2f is None else np.copy(self.f2f)
        pi_copy = None if self.pi is None else np.copy(self.pi)
        vta_copy = self.vta
        return Mesh(v_copy, f_copy, n_copy, f2f_copy, pi_copy, vta_copy)

    """
    Get methods
    Function to get information from the mesh.
    """

    # Calculates the center of the mesh by averaging all vertex positions.
    # Returns the center of the mesh.
    def getPCCenter(self):
        num_vertices = self.v.shape[0]
        center = np.sum(self.v, axis=0)/num_vertices
        return center

    # Calculates the size of the mesh.
    # The size is defined as the vertex with the maximum distance from the center of the mesh.
    # Returns 3x1 array, center of the mesh.
    def getPCSize(self):
        center = self.getPCCenter()
        return np.max(np.linalg.norm(self.v - np.tile(center, (self.v.shape[0], 1)), axis=1))
    
    # Calculates the bounding size in which the Mesh perfectly fits.
    # Returns 3x1 array containing the size of the bounding box.
    def getPCBoundingBox(self):
        return np.max(self.v, axis=0) - np.min(self.v, axis=0)

    # Calculates the areas of the given faces.
    # faces is an array with indices to faces from which the areas need to be calculated.
    # Returns #facesx1 array with the areas corresponding to the face indices.
    def getAreas(self, faces):
        triangles = self.v[self.f[faces]]
        As = triangles[:,1,:] - triangles[:,0,:]
        Bs = triangles[:,2,:] - triangles[:,0,:]
        cross = np.cross(As, Bs, axis=1)
        areas = 0.5*np.linalg.norm(cross, axis=1)
        return areas

    # Retrieve an array of face indices that contain vertices that are within range.
    # This method is described in the paper.
    # center is a 3x1 array representing the center of the sphere.
    # range is a number representing the radius of the sphere.
    # Returns an 1D array containing face indices of all faces that contain a vertex that is within the sphere.
    def getFacesInRange(self, center, range):
        translated = self.v - center
        dist = np.linalg.norm(translated, axis=1)
        vm = dist < range
        vertices = np.arange(vm.shape[0])[vm]
        faces = self.getTrianglesOfVertices(vertices)
        return faces

    
    # Gets the neighbouring faces as described by the paper.
    # face_index is the index of the face which is considered.
    # ring is a number (n) representing the size of the neighbourhood that needs to be returned.
    # Returns an array containing all faces within the n-ring neighbourhood.
    def getNeighbourhood(self, face_index, ring):
        adjecancy = self.f2f if not (self.f2f is None) else igl.triangle_triangle_adjacency(self.f)[0]
        self.f2f = adjecancy

        result = np.array(face_index)
        for _ in range(ring):
            nb = adjecancy[result]
            result = np.union1d(result, nb)

        return np.array(list(result))
    
    # Getter method for the face normals.
    # If they are not calculated yet, they are calculated upon calling this method.
    # Returns array of face normals.
    def getFaceNormals(self):
        if not (self.n is None):
            return self.n
        else:
            faceVertices = self.v[self.f]
            crosses = np.cross(faceVertices[:, 1, :] - faceVertices[:, 0, :], faceVertices[:, 2, :] - faceVertices[:, 1, :])
            faceNormals = crosses / np.linalg.norm(crosses, axis=1)[:, None]
            self.n = faceNormals
            return faceNormals
    
    # Getter method for getting the IGL vertex-triangle adjacency.
    # Sets the vta attribute.
    # Returns the adjacency matrix.
    def getVertexTriangleAdjacency(self):
        if self.vta is None:
            vta = igl.vertex_triangle_adjacency(self.f, len(self.v))
            self.vta = vta
            return vta
        else:
            return self.vta
    
    # Get the triangles which contain the given vertex index.
    # v is the vertex index.
    # Returns an array of face indices which contain the vertex v.
    def getTrianglesOfVertex(self, v):
        vta = self.getVertexTriangleAdjacency()
        triangles = vta[0][vta[1][v]:vta[1][v+1]]
        return triangles
    
    def getTrianglesOfVertices(self, vs):
        vta = self.getVertexTriangleAdjacency()
        start = vta[1][vs]
        stop = vta[1][vs+1]
        ranges = np.frompyfunc(np.arange, 2, 1)(start, stop)
        uniqueFaces = set()
        for range in ranges:
            uniqueFaces.update(vta[0][range])
        triangles = np.sort(np.array(list(uniqueFaces)))
        return triangles

    """
    Transformation methods
    These have functionality that transform the mesh.
    They return themselves, so that multiple transformations can be done in one line of code.
    """

    # Changes the model by translating / moving it in a certain direction.
    # translation is an 3x1 array representing the relative distance to move the mesh.
    # Returns the object itself
    def translate(self, translation):
        assert type(translation) == np.ndarray, "Translation should be a numpy array"
        assert translation.shape == (3,), f"Translation should be a 3 by 1 array, but is a {translation.shape} array"
        self.v += np.tile(translation, (self.v.shape[0], 1))
        return self

    # Resizes the mesh to the given size. Size is defined as the vertex with the maximum distance to the center of the mesh.
    # size is the target size to scale to.
    # Returns the object itself.
    def resize(self, size):
        assert type(size) == "float"
        center = self.getPCCenter()
        self.translate(-center)
        self.v = self.v/np.max(np.linalg.norm(self.v, axis=1))*size
        self.translate(center)
        return self

    # Rotates the mesh with a rotation matrix. This is basicly a matrix multiplication with all vertices.
    # rotationMatrix is a 3x3 array representing a rotation matrix.
    # Returns the object itself.
    def rotate(self, rotationMatrix):
        assert type(rotationMatrix) == np.ndarray and rotationMatrix.shape == (3, 3), "Rotation matrix is not a 3 by 3 matrix"
        center = self.getPCCenter()
        self.translate(-center)
        rotated_v = np.dot(rotationMatrix, self.v.T).T
        self.v = rotated_v
        self.translate(center)
        assert np.linalg.norm(center - self.getPCCenter()) < 0.001, f"Center of mesh was not the same after rotation, but shifted by {self.getPCCenter() - center}"
        return self

    """
    Show methods
    Shows visualizations of the mesh.
    """

    # Visualizes the mesh as a point cloud using meshplot.
    def mpShowPC(self):
        mp.plot(self.v, shading={"point_size": np.max(np.linalg.norm(self.v))/1000})
    
    # Visualizes the mesh a mesh using meshplot.
    def mpShowMesh(self):
        mp.plot(self.v, self.f, shading={"wireframe": True})

def psViewPC(v):
    ps.init()
    ps.register_point_cloud("main", v)
    ps.show()

def psViewMesh(v, f):
    ps.init()
    ps.register_surface_mesh("main", v, f)
    ps.show()

if __name__ == "__main__":
    myMesh = Mesh.readFile('MyProject/new_saved_fandisk.obj')
    patch = myMesh.getPatch(myMesh.v[0], 10)
    psViewPC(patch.v)