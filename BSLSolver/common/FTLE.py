from dolfin import *
from oasis.common import utilities

def ftle(mesh, V, u, original_bcs, dt, tstep, xdmf_f):
    new_bcs = []
    for i, bc in enumerate(original_bcs):
        subdomain = bc.user_sub_domain()
        new_bcs.append(DirichletBC(V, Constant(0), subdomain))

    CG1 = FunctionSpace(mesh, "CG", 1)
    #get the trajectories
    F = grad(u) + Identity(len(self.u))
    #calculate the right Cauchy Green Tensor
    C = F.T*F
    #Get the eigenvalues of the C tensor
    eigensolver = SLEPcEigenSolver(C)
    print("Computing eigenvalues of the Right Cauchy-Green Tensor")
    eigensolver.solve()
    #get the maximum eigenvalues
    eigens, _, _, _ = eigensolver.get_eigenpair(0) #not expecting any complex eigs
    #project the eigenvalues to a CG1 space
    eigs = OasisFunction(eigens, CG1, bcs=new_bcs, name='eigs',
                            method='default', solver_type='bicgstab',
                            preconditioner_type='jacobi')
    eigs()
    ftLe = 1/dt * ln(eigs**(1/2))
    
    #now just need to print to xdmffile
    xdmf_f.write(ftLe, float(tstep))
    



