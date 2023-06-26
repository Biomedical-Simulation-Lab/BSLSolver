import numpy as np
from scipy.special import cbrt
from dolfin import *
import warnings
warnings.filterwarnings('ignore')

def eigs_cardano(a, b, c, d):
    a,b,c,d = a+np.zeros(a.shape)*0.j, b+np.zeros(a.shape)*0.j, c+np.zeros(a.shape)*0.j, d+np.zeros(a.shape)*0.j
    Q = (3*a*c - b**2)/ (9*a**2)
    R = (9*a*b*c - 27*a**2*d - 2*b**3) / (54 * a**3)
    D = Q**3 + R**2

    numeric = np.isreal(R + np.sqrt(D))
    S = np.zeros(a.shape)
    S[numeric==True] = cbrt(np.real(R[numeric==True] + np.sqrt(D[numeric==True])))
    S[numeric==False] = (R[numeric==False] + np.sqrt(D[numeric==False]))**(1/3)

    T = np.zeros(a.shape)
    numeric2=np.isreal(R - np.sqrt(D))
    T[numeric2==True] = cbrt(np.real(R[numeric2==True] - np.sqrt(D[numeric2==True])))
    T[numeric2==False] = (R[numeric2==False] - np.sqrt(D[numeric2==False]))**(1/3)

    x1 = - b / (3*a) + (S+T)
    x2 = - b / (3*a)  - (S+T) / 2 + 0.5j * np.sqrt(3) * (S - T)
    x3 = - b / (3*a)  - (S+T) / 2 -  0.5j * np.sqrt(3) * (S - T)
    return np.array([x1, x2, x3])   

def ftle(mesh, V, u, original_bcs, dt, tstep, ftle_f):
    CG1 = FunctionSpace(mesh, "CG", 1)
    #get the trajectories
    def C(u):
        F = Identity(3) + grad(u)
        return F.T*F
    def C_b(u):
        F = Identity(3) - grad(u)
        return F.T*F
    
    def doubledot_trace(A):
        return A[0,0]**2+A[1,1]**2+A[2,2]**2
    #forward problem
    #get invariants
    C_ = C(u)
    D_ = doubledot_trace(C_)
    i1 = tr(C_)
    i2 = -0.5*(tr(C_)**2-D_)
    i3 = det(C_)
    I1 = project(i1, CG1).vector().get_local()
    I2 = project(i2, CG1).vector().get_local()
    I3 = project(i3, CG1).vector().get_local()

    if MPI.rank(MPI.comm_world) == 0:
        print("Computing eigenvalues of the Right Cauchy-Green Tensor for the forward problem")
    vals = eigs_cardano(-np.ones(I1.shape), I1, I2, I3)
    max_eig = Function(CG1)
    max_eig.vector().set_local(np.max(vals, axis = 0))

    ftLe_forward = project(1/dt * ln(max_eig**(1/2)),CG1)
    ftLe_forward.rename('ftLe_forward','forward-time')

    #backward problem
    #get invariants
    Cb_ = C_b(u)
    Db_ = doubledot_trace(Cb_)
    i1_b = tr(Cb_)
    i2_b = -0.5*(tr(Cb_)**2-Db_)
    i3_b = det(Cb_)
    I1_b = project(i1_b, CG1).vector().get_local()
    I2_b = project(i2_b, CG1).vector().get_local()
    I3_b = project(i3_b, CG1).vector().get_local()
    if MPI.rank(MPI.comm_world) == 0:
        print("Computing eigenvalues of the Right Cauchy-Green Tensor for the backward problem")
    vals_b = eigs(-np.ones(I1.shape), I1_b, I2_b, I3_b)
    max_eig_b = Function(CG1)
    max_eig_b.vector().set_local(np.max(vals_b, axis = 0))

    ftLe_backward = project(1/dt * ln(max_eig_b**(1/2)), CG1)
    ftLe_backward.rename('ftLe_backward','backward-time')
    
    #now just need to print to xdmffile
    with ftle_f as file:
        file.parameters.update({"rewrite_function_mesh": False})
        file.write(ftLe_forward, float(tstep))
        file.write(ftLe_backward, float(tstep))




