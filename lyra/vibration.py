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
    print_coordinate(atomic_number_i, coord_i, "Initial State Coordinate", f)
    print_coordinate(atomic_number_f, coord_f, "Final State Coordinate (Aligned)", f)

    hess_i = fchk_i.hessian()
    hess_f = fchk_f.hessian()
    xR = np.zeros((3*len(mass_f), 3*len(mass_f)))
    for i in range(len(mass_i)):
        xR[i*3:(i+1)*3, i*3:(i+1)*3] = R
    hess_f = xR @ hess_f @ xR.T

    freq_i, mwmodes_i = solve_hess(hess_i, mass_i, coord_i)
    freq_f, mwmodes_f = solve_hess(hess_f, mass_f, coord_f)

    Duschinsky = mwmodes_f.T @ mwmodes_i
    flatmass = np.repeat(mass_f, 3)
    dq = np.sqrt(flatmass) * (coord_i-coord_f).flat
    Displace = mwmodes_f.T @ dq

    S_f = 0.5 * freq_f * Displace**2
    lmd_f = freq_f * S_f
    print_vibronic_data(freq_i, freq_f, Displace, "Displacement (a.u.)", f)
    print_vibronic_data(freq_i, freq_f, S_f, "Huang-Rhys Factor", f)
    f.writelines("Total Reorganalization Energy (cm-1): {:.4f}".format(np.sum(lmd_f)/CM2AU))
    print_vibronic_data(freq_i, freq_f, lmd_f/CM2AU, "Reorganalization Energy (cm-1)", f)
    np.savetxt("Duschinsky.dat", Duschinsky, fmt="%10.6f")

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
    Displace = grad_Q_f / freq_i**2

    S = 0.5 * freq_i * Displace**2
    lmd = freq_i * S
    print_vibronic_data(freq_i, freq_i, Displace, "Displacement (a.u.)", f)
    print_vibronic_data(freq_i, freq_i, S, "Huang-Rhys Factor", f)
    f.writelines("Total Reorganalization Energy (cm-1): {:.4f}".format(np.sum(lmd)/CM2AU))
    print_vibronic_data(freq_i, freq_i, lmd/CM2AU, "Reorganalization Energy (cm-1)", f)

    f.close()
