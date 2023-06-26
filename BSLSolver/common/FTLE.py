import ufl
from dolfin import *
import warnings

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
    return val2

def ftle(mesh, u, dt, tstep, ftle_f):
    t = Timer()
    CG1 = FunctionSpace(mesh, "CG", 1)
    #get the trajectories
    def C(u):
        F = Identity(3) + grad(u)*dt
        return F.T*F
    def C_b(u):
        F = Identity(3) - grad(u)*dt
        return F.T*F
    
    def doubledot_trace(A):
        return A[0,0]**2+A[1,1]**2+A[2,2]**2
    #forward problem
    C_ = C(u)

    if MPI.rank(MPI.comm_world) == 0:
        print("Computing eigenvalues of the Right Cauchy-Green Tensor for the forward problem")
    vals = eigenstate(C_)
    max_eig = project(vals,CG1)
    ftLe_forward = project(1/dt * ln(max_eig**(1/2)),CG1)
    ftLe_forward.rename('ftLe_forward','forward-time')

    #backward problem
    Cb_ = C_b(u)
    if MPI.rank(MPI.comm_world) == 0:
        print("Computing eigenvalues of the Right Cauchy-Green Tensor for the backward problem")
    vals_b = eigenstate(Cb_)
    max_eig_b = project(vals_b,CG1)

    ftLe_backward = project(1/dt * ln(max_eig_b**(1/2)), CG1)
    ftLe_backward.rename('ftLe_backward','backward-time')
    if MPI.rank(MPI.comm_world) == 0:
        print('Finished finding eigenvalues in %f s'%t.elapsed()[0])
    
    #now just need to print to xdmffile
    with ftle_f as file:
        file.parameters.update({"rewrite_function_mesh": False})
        file.write(ftLe_forward, float(tstep))
        file.write(ftLe_backward, float(tstep))




