import numpy as np
from dolfin import assemble, interpolate, Measure, FacetNormal, Identity, VectorFunctionSpace, \
    BoundaryMesh, SubMesh, Function, FacetArea, TestFunction, FunctionSpace, inner, grad, sym
from BSLSolver.common import h5io

class STRESS:
    """Computes the stress on a given mesh based on provided velocity and pressure fields. This code is modified from nsbench and VAMPy."""

    def __init__(self, u, p, mu, mesh, fd):
        """Initializes the Stress class.

        Args:
            u (Function): The velocity field.
            p (Function): The pressure field.
            nu (float): The kinematic viscosity.
            mesh (Mesh): The mesh on which to compute stress.
        """
        boundary_ds = Measure("ds", domain=mesh)
        fd = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
        boundarymesh = BoundaryMesh(mesh, 'exterior')
        #have to map the fd MeshFunction that is defined on the mesh to the external boundary mesh
        bdim = boundarymesh.topology().dim()
        boundary_boundaries = MeshFunction('size_t', boundarymesh, bdim)
        boundary_boundaries.set_all(0)
        for i, facet in enumerate(entities(boundarymesh, bdim)): #go through all facets in the boundarymesh
            parent_meshentity = boundarymesh.entity_map(bdim)[i] #gives the index of the boundary facet in the parent mesh
            parent_boundarynumber = fd.array()[parent_meshentity] #search for the facet mesh_value_collection in the parent mesh
            boundary_boundaries.array()[i] = parent_boundarynumber #assigns the facet_mesh_value_collection number to the new MeshFunction 

        submesh_of_b0= SubMesh(boundarymesh, boundary_boundaries, 0) #fd=0 is the wall, so it should be same in the new submesh

        self.bmV = VectorFunctionSpace(submesh_of_b0, 'CG', 1)
        self.bMesh = submesh_of_b0

        # Compute stress tensor everywhere
        sigma = (2 * mu * self.epsilon(u)) - (p * Identity(len(u)))

        # Compute stress on mesh surfaces
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        Ft = F - (Fn * n)  # vector-valued

        # Integrate piecewise constants on the boundary
        scalar = FunctionSpace(mesh, 'DG', 0)
        vector = VectorFunctionSpace(mesh, 'CG', 1)
        scaling = FacetArea(mesh)  # Normalize the computed stress relative to the size of the element -- not used here

        v = TestFunction(scalar)
        w = TestFunction(vector)

        # Create function
        self.Ftv = Function(vector)

        # 1 / (2 * scaling) *
        self.Ltv = inner(w, Ft) * boundary_ds #L2 projection of DG0 stress (Ft) onto a CG1 mesh

    def call(self):
        """
        Compute stress for given velocity field u and pressure field p

        Returns:
            Ftv_bm (Function): Shear stress
        """
        self.Ftv_bm = Function(self.bmV, name='wss')
        # Assemble vectors
        assemble(self.Ltv, tensor=self.Ftv.vector())
        self.Ftv_bm = -interpolate(self.Ftv, self.bmV) #interpolate stress tensor onto the boundary only, but normals point out so have to add in the negative
        
        #get the cartesian norm of the wss
        Ftv_bm_abs = project(sqrt(inner(self.Ftv_bm,self.Ftv_bm)), self.bmV)
        Ftv_bm_abs.rename('wss_abs', 'absval')
        return self.Ftv_bm, Ftv_bm_abs, self.bMesh
    
    def norm_l2(self, u):
        """
        Compute norm of vector u in expression form
        Args:
            u (Function): Function to compute norm of

        Returns:
            norm (Power): Norm as expression
        """
        return pow(inner(u, u), 0.5)
    
    def epsilon(self):
        """
        Computes the strain-rate tensor
        """

        return sym(grad(self.u))

#compute the wall shear stress on the boundary using the vector u_
#STILL REQUIRES TESTING, but should be ok
def compute_wall_shear_stress(mesh, u_, nu, fd, results_folder, t, tstep, current_cycle, case_fullname):
    mu = nu*1057 #get dynamic viscosity using rho=1057
    case_fullname = case_fullname
    filepath = results_folder+'/wss_files/'+case_fullname+'_curcyc_%%d_t=%%0%d.4f_ts=%%0%dd_wss.h5'%(current_cycle, t, tstep)
    #Calculate stress
    tau, tau_abs, bmesh = STRESS(u_, 0.0, mu, mesh, fd)
    #Print to file
    meshpath = results_folder + '/wss_mesh.h5'
    if not meshpath.exists():
        fmesh = HDF5File(bmesh.mpi_comm(), meshpath, 'w')
        fmesh.write(bmesh, '/Mesh')
        normals = FacetNormal(bmesh)
        fmesh.write(normals, '/Mesh/normal')
    f = XDMFFile(filepath)
    f.parameters['rewrite_function_mesh'] = False #don't need the mesh
    f.write(tau, t)
    f.write(tau_abs, t)
    
