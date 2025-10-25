import numpy as np
import numpy.linalg as linalg


def move_to_origin(mass, coord):
    com = np.einsum("ij,i->j", coord, mass)/np.sum(mass)
    return coord - com


def get_principle_axes(mass, coord):
    inertia = -np.einsum("ki,k,kj->ij", coord, mass, coord)
    inertia -= np.eye(3) * np.trace(inertia)

    moments, axes = linalg.eigh(inertia)
    # Check if principal axes are right-handed
    # If not, then rearrange them to be right-handed
    axes[:, 2] *= 1 if linalg.det(axes) >= 0 else -1

    return moments, axes


def all_collinear(coord, already_noncollinear=False):
    GEOM_TOL = 1e-5

    if already_noncollinear:
        return False

    if coord.shape[0] == 2:
        return True
    else:
        v1 = coord[0, :] - coord[1, :]
        v2 = coord[2, :] - coord[1, :]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        cosval = np.dot(v1, v2)
        return all_collinear(coord[1:, :],
                             np.abs(cosval-1) > GEOM_TOL and
                             np.abs(cosval+1) > GEOM_TOL)


def eckart_projection(mass, coord):
    GEOM_TOL = 1e-5
    natoms = coord.shape[0]

    moments, axes = get_principle_axes(mass, coord)
    eckart_coord = coord @ axes

    T1 = np.zeros((natoms, 3))
    T2 = np.zeros((natoms, 3))
    T3 = np.zeros((natoms, 3))
    R1 = np.zeros((natoms, 3))
    R2 = np.zeros((natoms, 3))
    R3 = np.zeros((natoms, 3))
    for i in range(natoms):
        T1[i, 0] = np.sqrt(mass[i])
        T2[i, 1] = np.sqrt(mass[i])
        T3[i, 2] = np.sqrt(mass[i])
        R1[i, :] = np.sqrt(mass[i]) * (eckart_coord[i, 1] *
                                       axes[:, 2]-eckart_coord[i, 2]*axes[:, 1])
        R2[i, :] = np.sqrt(mass[i]) * (eckart_coord[i, 2] *
                                       axes[:, 0]-eckart_coord[i, 0]*axes[:, 2])
        R3[i, :] = np.sqrt(mass[i]) * (eckart_coord[i, 0] *
                                       axes[:, 1]-eckart_coord[i, 1]*axes[:, 0])
    T1 = T1.flat
    T2 = T2.flat
    T3 = T3.flat
    R1 = R1.flat
    R2 = R2.flat
    R3 = R3.flat

    criterion = all_collinear(coord) or np.abs(moments[0]) < GEOM_TOL

    L = np.zeros((5 if criterion else 6, 3*natoms))
    L[0, :] = T1/np.linalg.norm(T1)
    L[1, :] = T2/np.linalg.norm(T2)
    L[2, :] = T3/np.linalg.norm(T3)
    if not criterion:
        L[3, :] = R1/np.linalg.norm(R1)
    L[3 if criterion else 4, :] = R2/np.linalg.norm(R2)
    L[4 if criterion else 5, :] = R3/np.linalg.norm(R3)

    U, s, Vh = linalg.svd(L)

    return Vh[5 if criterion else 6:, :]


def geom_massweighted_rmsd(coord1, coord2, mass):
    rmsd = 0
    for i in range(len(mass)):
        dx = coord1[i, :] - coord2[i, :]
        rmsd += mass[i] * np.sum(dx*dx)
    return np.sqrt(rmsd/len(mass))


def kabsch_alignment(Q, P, mass):
    A = np.einsum("ki,k,kj->ij", P, mass, Q)
    U, s, Vh = linalg.svd(A)

    eyec = np.eye(3)
    eyec[2, 2] = 1 if linalg.det(U@Vh) >= 0 else -1

    R = U @ eyec @ Vh
    Pprime = P @ R

    return Pprime, geom_massweighted_rmsd(Pprime, Q, mass), R


def quaternions_alignment(Q, P, mass):
    Xp = np.einsum("i,ij->ij", np.sqrt(mass), Q + P)
    Xm = np.einsum("i,ij->ij", np.sqrt(mass), Q - P)

    M = np.zeros((4,4))
    from numpy import square as pow2
    M[0,0] = np.sum(pow2(Xm)) # xm2+ym2+zm2
    M[1,1] = np.sum(pow2(Xp[:,1:])) + np.sum(pow2(Xm[:,0])) # yp2+zp2+xm2
    M[2,2] = np.sum(pow2(Xp[:,0])) + np.sum(pow2(Xp[:,2])) + np.sum(pow2(Xm[:,1])) # xp2+zp2+ym2
    M[3,3] = np.sum(pow2(Xp[:,:2])) + np.sum(pow2(Xm[:,2])) # xp2+yp2+zm2
    M[1,0] = M[0,1] = np.dot(Xp[:,1],Xm[:,2]) - np.dot(Xp[:,2],Xm[:,1])
    M[2,0] = M[0,2] = np.dot(Xp[:,2],Xm[:,0]) - np.dot(Xp[:,0],Xm[:,2])
    M[3,0] = M[0,3] = np.dot(Xp[:,0],Xm[:,1]) - np.dot(Xp[:,1],Xm[:,0])
    M[2,1] = M[1,2] = np.dot(Xm[:,0],Xm[:,1]) - np.dot(Xp[:,0],Xp[:,1])
    M[3,1] = M[1,3] = np.dot(Xm[:,0],Xm[:,2]) - np.dot(Xp[:,0],Xp[:,2])
    M[3,2] = M[2,3] = np.dot(Xm[:,1],Xm[:,2]) - np.dot(Xp[:,1],Xp[:,2])

    l, q = linalg.eigh(M)
    q1, q2, q3, q4 = q[:,0]
    R = np.zeros((3,3))
    R[0,0] = q1*q1 + q2*q2 - q3*q3 - q4*q4
    R[1,1] = q1*q1 - q2*q2 + q3*q3 - q4*q4
    R[2,2] = q1*q1 - q2*q2 - q3*q3 + q4*q4
    R[1,0] = 2 * (q2*q3 + q1*q4)
    R[2,0] = 2 * (q2*q4 - q1*q3)
    R[2,1] = 2 * (q3*q4 + q1*q2)
    R[0,1] = 2 * (q2*q3 - q1*q4)
    R[0,2] = 2 * (q2*q4 + q1*q3)
    R[1,2] = 2 * (q3*q4 - q1*q2)

    Pprime = P @ R

    return Pprime, geom_massweighted_rmsd(Pprime, Q, mass), R
