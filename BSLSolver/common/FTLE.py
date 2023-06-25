from dolfin import *
from oasis.common import utilities

def ftle(mesh, V, u, original_bcs, dt, tstep, xdmf_f):
    new_bcs = []
    for i, bc in enumerate(original_bcs):
        subdomain = bc.user_sub_domain()
        new_bcs.append(DirichletBC(V, Constant(0), subdomain))

    CG1 = FunctionSpace(mesh, "CG", 1)
    #get the trajectories
    F = grad(u)*dt + Identity(len(u))
    #calculate the right Cauchy Green Tensor
    C = F.T*F
    #Get the eigenvalues of the C tensor
    eigensolver = SLEPcEigenSolver(C)
    print("Computing eigenvalues of the Right Cauchy-Green Tensor for the forward problem")
    eigensolver.solve()
    #get the maximum eigenvalues
    eigens, _, _, _ = eigensolver.get_eigenpair(0) #not expecting any complex eigs
    #project the eigenvalues to a CG1 space
    eigs = OasisFunction(eigens, CG1, bcs=new_bcs, name='eigs',
                            method='default', solver_type='bicgstab',
                            preconditioner_type='jacobi')
    eigs()
    ftLe_forward = 1/dt * ln(eigs**(1/2))
    ftLe_forward.rename('ftLe_forward','forward-time')

    F_b = Identity(len(u))- grad(u)*dt
    #calculate the right Cauchy Green Tensor
    C_b = F_b.T*F_b
    #Get the eigenvalues of the C tensor
    eigensolver_b = SLEPcEigenSolver(C_b)
    print("Computing eigenvalues of the Right Cauchy-Green Tensor fro the backward problem")
    eigensolver_b.solve()
    #get the maximum eigenvalues
    eigens_b, _, _, _ = eigensolver.get_eigenpair(0) #not expecting any complex eigs
    #project the eigenvalues to a CG1 space
    eigs_b = OasisFunction(eigens_b, CG1, bcs=new_bcs, name='eigs_b',
                            method='default', solver_type='bicgstab',
                            preconditioner_type='jacobi')
    eigs_b()
    ftLe_backward = 1/dt * ln(eigs_b**(1/2))
    ftLe_backward.rename('ftLe_backward','backward-time')
    
    #now just need to print to xdmffile
    with XDMFFile(MPI.comm_world, 'xdmf_f.xdmf') as file:
        file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False
        })
        file.write(ftLe_forward, float(tstep))
        file.write(ftLe_backward, float(tstep))




