import numpy as np
import numpy.linalg as linalg

from lyra.data import AMU2AU, CM2AU, MASSES
from lyra.utils import Formchk, print_coordinate, print_vibronic_data
from lyra.geom import move_to_origin, eckart_projection, kabsch_alignment


def solve_hess(hess, mass, coord):
    P = eckart_projection(mass, coord)
    flatmass = np.repeat(mass, 3)
    invsqrt_mass = 1/np.sqrt(flatmass)
    mwhess = np.einsum("i,ij,j->ij", invsqrt_mass, hess, invsqrt_mass)

    in_mwhess = P @ mwhess @ P.T
    freq, in_mwmodes = linalg.eigh(in_mwhess)
    freq = np.piecewise(freq, [freq > 0, freq < 0],
                        [lambda x: np.sqrt(x), lambda x: -np.sqrt(-x)])
    mwmodes = P.T @ in_mwmodes

    return freq, mwmodes


def apply_rotation_on_hess(hess, R):
    xR = np.zeros_like(hess)
    for i in range(hess.shape[0]//3):
        xR[i*3:(i+1)*3, i*3:(i+1)*3] = R
    return xR @ hess @ xR.T


def calculate_frequency(fchk_file):
    fchk = Formchk(fchk_file)

    atomic_number = fchk.key_to_value("Atomic numbers").astype(int)
    mass = np.array([MASSES[n]*AMU2AU for n in atomic_number])
    coord = fchk.coordinate()
    hess = fchk.hessian()

    coord = move_to_origin(mass, coord)
    freq, mwmodes = solve_hess(hess, mass, coord)
    flatmass = np.repeat(mass, 3)
    invsqrt_mass = 1/np.sqrt(flatmass)
    modes = np.einsum("i,ij->ij", invsqrt_mass, mwmodes)
    for i in range(modes.shape[1]):
        modes[:, i] /= linalg.norm(modes[:, i])

    return freq, modes


def electronic_vibrational_AH(fchk_file_i, fchk_file_f):
    fchk_i = Formchk(fchk_file_i)
    fchk_f = Formchk(fchk_file_f)

    atomic_number_i = fchk_i.key_to_value("Atomic numbers").astype(int)
    atomic_number_f = fchk_f.key_to_value("Atomic numbers").astype(int)
    assert (np.all(atomic_number_i == atomic_number_f))

    mass_i = np.array([MASSES[n]*AMU2AU for n in atomic_number_i])
    mass_f = np.array([MASSES[n]*AMU2AU for n in atomic_number_f])
    coord_i = fchk_i.coordinate()
    coord_f = fchk_f.coordinate()
    coord_i = move_to_origin(mass_i, coord_i)
    coord_f = move_to_origin(mass_f, coord_f)

    coord_f, rmsd, R = kabsch_alignment(coord_i, coord_f, mass_f)

    f = open("lyra_vibronic_analysis.out", 'w')
    print_coordinate(atomic_number_i, coord_i, f'Initial State Coordinate "{fchk_file_i}"', f)
    print_coordinate(atomic_number_f, coord_f, f'Final State Coordinate (Aligned) "{fchk_file_f}"', f)

    hess_i = fchk_i.hessian()
    hess_f = fchk_f.hessian()
    hess_f = apply_rotation_on_hess(hess_f, R)

    freq_i, mwmodes_i = solve_hess(hess_i, mass_i, coord_i)
    freq_f, mwmodes_f = solve_hess(hess_f, mass_f, coord_f)

    def Duschinsky(xyz1, xyz2, mwmode1, mwmode2, freq1, mass):
        J = mwmode1.T @ mwmode2
        flatmass = np.repeat(mass, 3)
        dQcart = np.sqrt(flatmass) * (xyz1-xyz2).flat
        dQ = mwmode1.T @ dQcart

        S = 0.5 * freq1 * dQ**2
        lmd = freq1 * S

        return J, dQ, S, lmd

    J_if, dQ_if, S_i, lmd_i = Duschinsky(coord_i, coord_f, mwmodes_i, mwmodes_f, freq_i, mass_i)
    J_fi, dQ_fi, S_f, lmd_f = Duschinsky(coord_f, coord_i, mwmodes_f, mwmodes_i, freq_f, mass_f)

    def print_vibronic_analysis(f, ref, freq, dQ, S, lmd):
        f.writelines(f'Initial State "{ref}" as the Reference State\n')
        print_vibronic_data(freq, dQ, "Displacement (a.u.)", f)
        print_vibronic_data(freq, S, "Huang-Rhys Factor", f)
        f.writelines("Total Reorganization Energy (cm-1): {:.4f}\n".format(np.sum(lmd)/CM2AU))
        print_vibronic_data(freq, lmd/CM2AU, "Reorganization Energy (cm-1)", f)
        f.writelines("\n")

    print_vibronic_analysis(f, fchk_file_i, freq_i, dQ_if, S_i, lmd_i)
    print_vibronic_analysis(f, fchk_file_f, freq_f, dQ_fi, S_f, lmd_f)

    np.savetxt("Duschinsky_if.dat", J_if, fmt="%10.6f")
    np.savetxt("Duschinsky_fi.dat", J_fi, fmt="%10.6f")

    f.close()


def electronic_vibrational_VG(fchk_file_i, fchk_file_f):
    fchk_i = Formchk(fchk_file_i)
    fchk_f = Formchk(fchk_file_f)

    atomic_number_i = fchk_i.key_to_value("Atomic numbers").astype(int)
    atomic_number_f = fchk_f.key_to_value("Atomic numbers").astype(int)
    assert (np.all(atomic_number_i == atomic_number_f))

    mass_i = np.array([MASSES[n]*AMU2AU for n in atomic_number_i])
    mass_f = np.array([MASSES[n]*AMU2AU for n in atomic_number_f])
    coord_i = fchk_i.coordinate()
    coord_i = move_to_origin(mass_i, coord_i)

    hess_i = fchk_i.hessian()
    grad_f = fchk_f.grad()

    freq_i, mwmodes_i = solve_hess(hess_i, mass_i, coord_i)

    f = open("lyra_vibronic_analysis.out", 'w')
    print_coordinate(atomic_number_i, coord_i, "Initial State Coordinate", f)

    invsqrt_mass = 1/np.sqrt(np.repeat(mass_f, 3))
    grad_Q_f = np.einsum("j,j,ji->i", grad_f.flat, invsqrt_mass, mwmodes_i)
    dQ = grad_Q_f / freq_i**2

    S = 0.5 * freq_i * dQ**2
    lmd = freq_i * S
    g = grad_Q_f / np.sqrt(2*freq_i**3)
    print_vibronic_data(freq_i, freq_i, dQ, "Displacement (a.u.)", f)
    print_vibronic_data(freq_i, freq_i, g, "Dimensionless Displacement", f)
    print_vibronic_data(freq_i, freq_i, S, "Huang-Rhys Factor", f)
    f.writelines("Total Reorganization Energy (cm-1): {:.4f}".format(np.sum(lmd)/CM2AU))
    print_vibronic_data(freq_i, freq_i, lmd/CM2AU, "Reorganization Energy (cm-1)", f)

    f.close()
