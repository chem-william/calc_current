# This code was tested using Python 3.6.9, GPAW 19.8.1, and ASE 3.19.0
import os.path

from ase import Atoms
from ase.io import read
from ase.io import write
from ase.visualize import view
import numpy as np


def align_end(indices, Mol, A, D, Alkyne=False):
    SH_dist = 2  # Use 1.75 for -SH2
    HH_dist = 0.75

    Ha = Atoms(
        ['H'],
        [Mol[indices['S_ind']].position + [0, 0, SH_dist]]
    )
    Hb = Atoms(
        ['H'],
        [Mol[indices['S_ind']].position + [0, 0, SH_dist + HH_dist]]
    )

    Mol += Ha + Hb

    Mol.set_angle(indices['C_ind'], indices['S_ind'], -2, A)
    Mol.set_angle(indices['C_ind'], indices['S_ind'], -1, A)

    if Alkyne is False:
        Mol.set_dihedral(
            indices['Si_ind'],
            indices['C_ind'],
            indices['S_ind'], -2, D
        )

        Mol.set_dihedral(
            indices['Si_ind'],
            indices['C_ind'],
            indices['S_ind'], -1, D
        )

    else:
        for n in np.arange(0, 360, 0.1):
            old1 = Mol.get_distance(Ha, -1)
            old2 = Mol.get_distance(Hb, -1)

            Mol.set_dihedral(
                indices['Si_ind'],
                indices['C_ind'],
                indices['S_ind'], -2, n
            )

            Mol.set_dihedral(
                indices['Si_ind'],
                indices['C_ind'],
                indices['S_ind'], -1, n
            )

            if (
                    Mol.get_distance(Ha, -1) > Mol.get_distance(Hb, -1)
                    and old1 < old2
            ):
                break

            print('dihedral is ', n)


def create_junction(Mol, path, A, D1, D2):
    # Define vacuum indices
    start_indices = {
        'S_ind': 9,
        'C_ind': 6,
        'Si_ind': 12,
    }

    end_indices = {
        'S_ind': 16,
        'C_ind': 0,
        'Si_ind': 19,
    }

    # align S-C with y-axis
    CCvec = Mol[start_indices['C_ind']].position - Mol[end_indices['C_ind']].position
    Mol.rotate(CCvec, 'z')

    # Align both ends
    align_end(start_indices, Mol, A, D1)
    align_end(end_indices, Mol, A, D2)

    # set cell og periodic boundary conditions
    Mol.center(vacuum=4.0)

    write('{}/hh_junc.traj'.format(path),  Mol)
    return Mol


if __name__ == '__main__':
    for tmp_file in os.listdir("./"):
        if tmp_file.endswith(".xyz"):
            xyz_file = tmp_file

    mol = read(xyz_file)

    a = create_junction(mol, "./", 120, 120, 120)
    for atom in a:
        print(atom)

    view(a)
