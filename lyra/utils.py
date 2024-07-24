import numpy as np
from lyra.data import CM2AU, ANG2AU, ELEMENTS


class Formchk:
    def __init__(self, file_path):
        self.file_path = file_path
        self.natm = NotImplemented
        self.nao = NotImplemented
        self.nmo = NotImplemented
        self.initialization()

    def initialization(self):
        self.natm = int(self.key_to_value("Number of atoms"))
        self.nao = int(self.key_to_value("Number of basis functions"))
        self.nmo = int(self.key_to_value("Number of independent functions"))

    def key_to_value(self, key, file_path=None):
        if file_path is None:
            file_path = self.file_path
        flag_read = False
        expect_size = -1
        vec = []
        with open(file_path, "r") as file:
            for l in file:
                if l[:len(key)] == key:
                    try:
                        expect_size = int(l[len(key):].split()[2])
                        flag_read = True
                        continue
                    except IndexError:
                        try:
                            return float(l[len(key):].split()[1])
                        except IndexError:
                            continue
                if flag_read:
                    try:
                        vec += [float(i) for i in l.split()]
                    except ValueError:
                        break
        if len(vec) != expect_size:
            raise ValueError(
                "Number of expected size is not consistent with read-in size!")
        return np.array(vec)

    def total_energy(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Total Energy", file_path)

    def coordinate(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Current cartesian coordinates", file_path).reshape((self.natm, 3))

    def grad(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Cartesian Gradient", file_path).reshape((self.natm, 3))

    def dipole(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Moment", file_path)

    @staticmethod
    def tril_to_symm(tril: np.ndarray):
        dim = int(np.floor(np.sqrt(tril.size * 2)))
        if dim * (dim + 1) / 2 != tril.size:
            raise ValueError("Size " + str(tril.size) +
                             " is probably not a valid lower-triangle matrix.")
        indices_tuple = np.tril_indices(dim)
        iterator = zip(*indices_tuple)
        symm = np.empty((dim, dim))
        for it, (row, col) in enumerate(iterator):
            symm[row, col] = tril[it]
            symm[col, row] = tril[it]
        return symm

    def hessian(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.tril_to_symm(self.key_to_value("Cartesian Force Constants", file_path))

    def polarizability(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        return self.tril_to_symm(self.key_to_value("Polarizability  ", file_path))

    def dipolederiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Derivatives", file_path).reshape(-1, 3)


def write_freq_modes(freq, modes):

    def write_chunks(array, file):
        n = len(array)
        nchunk = n//5
        nrem = n - 5 * nchunk

        for i in range(nchunk):
            for j in range(5):
                file.write("{:>16.8e}".format(array[i*5+j]))
            file.write("\n")
        for i in range(nrem):
            file.write("{:>16.8e}".format(array[nchunk*5+i]))
        file.write("\n")

    f = open("lyra_freq_modes.out", 'w')
    write_chunks(freq/CM2AU, f)
    for i in range(modes.shape[1]):
        write_chunks(modes[:, i], f)
    f.close()


def print_coordinate(atomic_number, coord, title, file):
    file.writelines(title)
    file.writelines('''
---------------------------------------------
                Coordinates (Angstroms)
  Atom         X           Y           Z
---------------------------------------------
''')
    for i in range(len(atomic_number)):
        file.writelines("{:>4s}{:>16.6f}{:>12.6f}{:>12.6f}\n".format(
            ELEMENTS[atomic_number[i]], 
            coord[i, 0]/ANG2AU, coord[i, 1]/ANG2AU, coord[i, 2]/ANG2AU))
    file.writelines("---------------------------------------------\n\n")


def print_vibronic_data(freq_i, freq_f, data, title, file):
    file.writelines('''
------------------------------------------
           Frequency (cm-1)
''')
    file.writelines("   No.    Initial     Final   {}\n".format(title))
    file.writelines("---------------------------------------------\n")
    for i in range(len(freq_f)):
        file.write("{:>5d}{:>11.2f}{:>11.2f}{:>12.4f}\n".format(
            i+1, freq_i[i]/CM2AU, freq_f[i]/CM2AU, data[i]))
    file.writelines("---------------------------------------------\n\n")

