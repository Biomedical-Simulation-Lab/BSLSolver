import ufl
from dolfin import *
from oasis.common import utilities
import warnings

_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi')

components = ['x','y','z']

def eigenstate(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.
    Note: Tensor A must not have complex eigenvalues!
    """
    eps = 1.0e-10
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = ufl.tr(A) / 3
    B = A - q * ufl.Identity(3)
    # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
    j = ufl.tr(B * B) / 2  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B * B * B) / 3  # == I3(B) for trace-free B
    # solve: 0 = ω**3 - j * ω - b  by substitution  ω = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4 / 3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2 / ufl.sqrt(3) * ufl.sqrt(j + eps ** 2)  # eps: MMM
    r = 4 * b / p ** 3
    r = ufl.Max(ufl.Min(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
    phi = ufl.acos(r) / 3
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    val0 = q + p * ufl.cos(phi + 2 / 3 * ufl.pi)  # low
    val1 = q + p * ufl.cos(phi + 4 / 3 * ufl.pi)  # middle
    val2 = q + p * ufl.cos(phi)  # high

    E0 = ufl.diff(val0, A).T
    E1 = ufl.diff(val1, A).T
    E2 = ufl.diff(val2, A).T
    #
    return val2, E0

def setup_ftle(mesh, V, u, dt):
    #get the trajectories
    def C(u):
        F = Identity(3) + grad(u)*dt
        return F.T*F
    def C_b(u):
        F = Identity(3) - grad(u)*dt
        return F.T*F
    #forward problem
    C_ = C(u)
    vals, _ = eigenstate(C_)
    #max_eig = project(vals,CG1)
    ftLe_forward = utilities.CG1Function(1/dt * ln(vals**(1/2)), mesh, method=_krylov_solver, name="ftLe_forward")
    #backward problem
    Cb_ = C_b(u)
    vals_b, _ = eigenstate(Cb_)
    #max_eig_b = project(vals_b,CG1)
    ftLe_backward = utilities.CG1Function(1/dt * ln(vals_b**(1/2)), mesh, method=_krylov_solver, name="ftLe_backward")
    ftLe_intersect = utilities.CG1Function(1/dt * (ln(vals**(1/2))-ln(vals_b**(1/2))), mesh, method=_krylov_solver, name="ftLe_intersect")
    #set up a library of gradients of sigma (backward)
    grad_sig = {ui: utilities.GradFunction(ftle_backward, V, i=i, name='dsigd' + ('x', 'y', 'z')[i], method=_krylov_solver) for i, ui in enumerate(components)} #not vectorized yet
    return ftLe_backward, ftLe_forward, ftLe_intersect, grad_sig

def get_ftle(ftLe_backward, ftLe_forward, ftLe_intersect, grad_sig, mesh, ftle_f, tstep):
    t = Timer()
    ftLe_forward()
    ftLe_backward()
    ftLe_intersect()
    if MPI.rank(MPI.comm_world) == 0:
        print('Finished finding ftLe fields in %f s'%t.elapsed()[0])
        
    #get Hessian matrix of backward (attracting ftle)
    _grad_sig  = as_vector([grad_sig[ui](ftLe_backward) for ui in components])
    hess = grad(_grad_sig) #ufl Hessian matrix DG0
    #get minimum eigenvector
    _, e_min_DG0 = eigenstate(hess) #the Hessian should always have real eigenvalues for any real function such as the ftle field
    e_min = utilities.CG1Function(e_min_DG0, mesh, method=_krylov_solver, name="e_min") #project to CG1
    e_min()
    #get the minima
    lcs = utilities.CG1Function(inner(grad_sig, e_min), mesh, method=_krylov_solver, name="lcs") #scalar-valued    
    lcs()
    if MPI.rank(MPI.comm_world) == 0:
        print('Finished finding total LCS in %f s'%t.elapsed()[0])
    #now just need to print to xdmffile
    with ftle_f as file:
        file.parameters.update({"rewrite_function_mesh": False})
        file.write(ftLe_forward, float(tstep))
        file.write(ftLe_backward, float(tstep))
        file.write(ftLe_intersect, float(tstep))
        file.write(lcs, float(tstep))




