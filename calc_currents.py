"""
This code was tested using Python 3.6.4, GPAW 1.5.1, and ASE 3.16.0
This script must be run with access to utils_zcolor.py and with an appropriately setup 'data' directory which can be generated with the make_dirs.sh shell script.
The output is a jmol (.spt) script, and can be collected with the collect_spt.py script.

For help contact marc@chem.ku.dk / marchamiltongarner@hotmail.com

Please cite the appropriate work.
s-band electrode transmission code: DOI: 10.1021/acs.jpclett.8b03432
Current density code: Jensen et al. "When Current Does Not Follow Bonds: Current Density In Saturated Molecules" Submitted 2018
Current density with cylindrical coordinates: Garner et al. "Helical Orbitals and Circular Currents in Linear Carbon Wires" Submitted 2018
"""

import argparse
import os
import sys
import pickle
from ase import Atoms
from ase.io import read
from ase.io import write
from ase.units import Hartree
from gpaw import FermiDirac
from gpaw import GPAW
from gpaw.lcao.tools import dump_hamiltonian_parallel
from gpaw.lcao.tools import get_bfi
from numpy import ascontiguousarray as asc
import gpaw
import matplotlib
import numpy as np
# To retrieve path to utils_zcolor which should be placed in the same folder as
# calc_current.py
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/bin/py_scripts/calc_current')
import utils_zcolor
from libgradient import Gradient

matplotlib.use('agg')


def read_config(config_file):
    with open(config_file) as file:
        lines = file.readlines()
        values = [line.split('=') for line in lines]
        config_values = {}
        for value in values:
            config_values[value[0]] = value[1].strip()

    return config_values


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--path',
        default='data/',
        help='path to data folder'
    )
    parser.add_argument(
        '--basis',
        default='dzp',
        help='basis (sz, dzp, ...)'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='name of the config file'
    )
    args = parser.parse_args()
    path = str(os.path.abspath(args.path)) + "/"

    basis = args.basis
    xyzname = "hh_junc.traj"
    config = args.config

    # Load config values (e.g. h-spacing, functional, align atoms etc.)
    if config is None:
        raise FileNotFoundError('No config file has been chosen')

    config_values = read_config(config)

    # Configuration and constants
    if "ef" in config_values:
        ef = float(config_values["ef"])
    else:
        ef = 0.0

    if 'functional' in config_values:
        xc = config_values['functional']

        # To make sure the right basis set is used
        basis += "." + xc
    else:
        xc = 'PBE'

    h_basis = "sz." + xc

    if 'h_spacing' in config_values:
        h = float(config_values['h_spacing'])
    else:
        h = 0.2

    if 'charge' in config_values:
        charge = int(config_values['charge'])
    else:
        charge = 0

    if "cutoff" in config_values:
        cutoff = int(config_values["cutoff"])
    else:
        cutoff = 20

    # Constants
    FDwidth = 0.1
    kpts = (1, 1, 1)
    mode = 'lcao'
    eV2au = 1/Hartree
    bias = 1e-3 * eV2au
    gamma = 1e0 * eV2au
    ef = ef * eV2au  # If ef has been set to a custom value.
    estart, eend = [-6 * eV2au, 6 * eV2au]
    energy_step = 1e-2 * eV2au
    energy_grid = np.arange(estart, eend, energy_step)

    # Align z-axis and cutoff at these atoms, OBS paa retningen.
    if 'bottom_atom' not in config_values:
        raise ValueError('You need to specify the bottom atom to align molecule')

    if 'top_atom' not in config_values:
        raise ValueError('You need to specify the top atom to align molecule')

    align1 = int(config_values['bottom_atom']) - 1
    align2 = int(config_values['top_atom']) - 1

    basis_full = {
        'H': h_basis,
        'C': basis,
        'S': basis,
        'N': basis,
        'Si': basis,
        'Ge': basis,
        'B': basis,
        'O': basis,
        'F': basis,
        'Cl': basis,
        'P': basis,
        'Br': basis,
        'Ru': basis
    }

    # Divider for h - size of the arrows for current density
    grid_size = 1

    plot_basename = "plots/"
    data_basename = "data/"

    molecule = read(path + xyzname)

    # Identify end atoms and align according to z-direction
    atoms = utils_zcolor.identify_and_align(molecule, align1, align2)
    symbols = atoms.get_chemical_symbols()
    np.save(path + "positions.npy", atoms.get_positions())
    np.save(path + "symbols.npy", symbols)
    atoms.write(path + "central_region.xyz")

    # Run and converge calculation
    calc = GPAW(
        h=h,
        xc=xc,
        basis=basis_full,
        occupations=FermiDirac(width=FDwidth),
        kpts=kpts,
        mode=mode,
        nbands="nao",
        symmetry={'point_group': False, 'time_reversal': False},
        charge=charge,
        txt="logfile.txt"
    )
    atoms.set_calculator(calc)
    atoms.get_potential_energy()  # Converge everything!
    print('Fermi is', atoms.calc.get_fermi_level())

    dump_hamiltonian_parallel(path + 'scat_', atoms, direction='z')

    # Write AO basis to disk
    bfs = get_bfi(calc, range(len(atoms)))
    rot_mat = np.diag(v=np.ones(len(bfs)))
    c_fo_xi = asc(rot_mat.real.T)  # coefficients
    phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
    gd0 = calc.wfs.gd

    calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, phi_xg, -1)
    # Writing this to disk while going multiprocessing with GPAW
    # Reason: Numpy tries to pickle the objects, but gd0 is 
    # not pickleable as it's a MPI object
    # np.save(path + "ao_basis_grid", [phi_xg, gd0])
    utils_zcolor.plot_basis(atoms, phi_xg, folder_name=path + "basis/ao")

    H_ao, S_ao = pickle.load(open(path + 'scat_0.pckl', 'rb'))
    H_ao = H_ao[0, 0]
    H_ao *= eV2au
    S_ao = S_ao[0]

    # Convert AO's to MO's
    eig_vals, eig_vec = np.linalg.eig(np.dot(np.linalg.inv(S_ao), H_ao))
    order = np.argsort(eig_vals)
    eig_vals = eig_vals.take(order)
    eig_vec = eig_vec.take(order, axis=1)
    S_mo = np.dot(np.dot(eig_vec.T.conj(), S_ao), eig_vec)
    eig_vec = eig_vec / np.sqrt(np.diag(S_mo))
    S_mo = np.dot(np.dot(eig_vec.T.conj(), S_ao), eig_vec)
    H_mo = np.dot(np.dot(eig_vec.T, H_ao), eig_vec)

    # Save the MO basis as .cube files
    c_fo_xi = asc(eig_vec.real.T)  # coefficients
    mo_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
    calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, mo_phi_xg, -1)
    np.save(path + "mo_energies", eig_vals)
    # np.save(path + "mo_basis", mo_phi_xg)
    utils_zcolor.plot_basis(atoms, mo_phi_xg, folder_name=path + "basis/mo")
    np.save(path + "mo_basis.npy", mo_phi_xg)

    n = len(H_ao)

    GamL = np.zeros([n, n])
    GamR = np.zeros([n, n])
    GamL[0, 0] = gamma
    GamR[n - 1, n - 1] = gamma

    Gamma_L = [GamL for en in range(len(energy_grid))]
    Gamma_R = [GamR for en in range(len(energy_grid))]
    Gamma_L = np.swapaxes(Gamma_L, 0, 2)
    Gamma_R = np.swapaxes(Gamma_R, 0, 2)

    Gr = utils_zcolor.ret_gf_ongrid(energy_grid, H_ao, S_ao, Gamma_L, Gamma_R)
    
    # Calculate the transmission - AO basis
    print("Calculating transmission - AO-basis")

    # To optimize the following matrix operations
    Gamma_L = Gamma_L.astype(dtype='float32', order='F')
    Gamma_R = Gamma_R.astype(dtype='float32', order='F')
    
    Gr = Gr.astype(dtype='complex64', order='F')
    trans = utils_zcolor.calc_trans(energy_grid, Gr, Gamma_L, Gamma_R)

    utils_zcolor.plot_transmission(
        energy_grid*Hartree, np.real(trans), path + plot_basename + "trans.png"
    )
    np.save(
        path + data_basename + 'trans_full.npy', [energy_grid*Hartree, trans]
    )
    print("AO-transmission done!")

    # Find HOMO and LUMO
    for n in range(len(eig_vals)):
        if eig_vals[n] < 0 and eig_vals[n + 1] > 0:
            HOMO = eig_vals[n]
            LUMO = eig_vals[n + 1]
            midgap = (HOMO + LUMO)/2.0

            np.savetxt(
                path + "basis/mo/" + 'homo_index.txt',
                X=['HOMO index is ', n],
                fmt='%.10s',
                newline='\n',
            )
            break

    hl_gap = [
        'HOMO is ',
        HOMO*Hartree,
        'LUMO is ',
        LUMO*Hartree,
        'mid-gap is ',
        midgap*Hartree
    ]
    np.savetxt(path + 'HOMO_LUMO.txt', X=hl_gap, fmt='%.10s', newline='\n')


    """Current with fermi functions"""
    fR, fL = utils_zcolor.fermi_ongrid(energy_grid, ef, bias)
    dE = energy_grid[1] - energy_grid[0]
    current_trans = (1 / (2*np.pi)) * np.array(
        [trans[en].real * (fL[en]-fR[en])*dE for en in range(len(energy_grid))]
    ).sum()
    np.save(path + data_basename + "current_trans.npy", current_trans)


    """Current approx at low temp"""
    sigma_r = -1j/2. * (GamL + GamR)  # + V_pot
    Gr_approx = utils_zcolor.retarded_gf2(H_ao, S_ao, ef, sigma_r)

    Gles = np.dot(np.dot(Gr_approx, GamL), Gr_approx.T.conj())
    Gles *= bias
    np.save(path + data_basename + "matrices_dV.npy", [Gr_approx, Gles, GamL])

    utils_zcolor.plot_complex_matrix(Gles, path + "Gles")

    Tt = np.dot(np.dot(np.dot(GamL, Gr_approx), GamR), Gr_approx.T.conj())
    current_dV = (bias/(2*np.pi))*Tt.trace()
    np.save(path + "data/trans_dV.npy", [ef, Tt.trace()])
    np.save(path + "data/current_dV.npy", current_dV)

    # Non corrected current - AO basis
    mlt = 1j*Gles/(4*np.pi)
    np.save(path + "data/Gles_dV.npy", mlt)

    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)
    np.save(path + "data/xyz_cor.npy", [x_cor, y_cor, z_cor])

    dx = x_cor[1] - x_cor[0]
    dy = y_cor[1] - y_cor[0]
    dz = z_cor[1] - z_cor[0]

    # MO - basis
    eig, vec = np.linalg.eig(np.dot(np.linalg.inv(S_ao), H_ao))
    order = np.argsort(eig)
    eig = eig.take(order)
    vec = vec.take(order, axis=1)
    S_mo = np.dot(np.dot(vec.T.conj(), S_ao), vec)
    vec = vec/np.sqrt(np.diag(S_mo))
    S_mo = np.dot(np.dot(vec.T.conj(), S_ao), vec)
    H_mo = np.dot(np.dot(vec.T, H_ao), vec)

    GamL_mo = np.dot(np.dot(vec.T, GamL), vec)
    GamR_mo = np.dot(np.dot(vec.T, GamR), vec)
    Gamma_L_mo = [GamL_mo for en in range(len(energy_grid))]
    Gamma_R_mo = [GamR_mo for en in range(len(energy_grid))]
    Gamma_L_mo = np.swapaxes(Gamma_L_mo, 0, 2)
    Gamma_R_mo = np.swapaxes(Gamma_R_mo, 0, 2)

    # Convert and save mlt in MO basis
    sigma_r_mo = -1j/2. * (GamL_mo + GamR_mo)  # + V_pot
    Gr_approx_mo = utils_zcolor.retarded_gf2(H_mo, S_mo, ef, sigma_r_mo)

    Gles_mo = np.dot(np.dot(Gr_approx_mo, GamL_mo), Gr_approx_mo.T.conj())
    Gles_mo *= bias
    np.save(path + data_basename + "matrices_dV_mo.npy", [Gr_approx_mo, Gles_mo, GamL_mo])

    utils_zcolor.plot_complex_matrix(Gles_mo, path + "Gles_mo")

    Tt_mo = np.dot(np.dot(np.dot(GamL_mo, Gr_approx_mo), GamR_mo), Gr_approx_mo.T.conj())
    current_dV_mo = (bias/(2*np.pi))*Tt_mo.trace()
    np.save(path + "data/trans_dV_mo.npy", [ef, Tt_mo.trace()])
    np.save(path + "data/current_dV_mo.npy", current_dV_mo)

    # Non corrected current - MO basis
    mlt_mo = 1j*Gles_mo/(4*np.pi)
    np.save(path + "data/Gles_dV_mo.npy", mlt_mo)

    print("Calculating gradient..")
    print("Calculating real part of current")
    current_c, jx, jy, jz = Gradient().jc_current(mo_phi_xg, mlt_mo.real, dx, dy, dz)
    dims = phi_xg.shape
    jx = np.reshape(jx, dims[1:])
    jy = np.reshape(jy, dims[1:])
    jz = np.reshape(jz, dims[1:])

    np.save(
        path + "data/" + "current_all.npy",
        np.array([jx, jy, jz, x_cor, y_cor, z_cor])
    )
    np.save(
        path + data_basename + "current_c.npy",
        np.array([current_c])
    )

    SI = 31
    EI = -31
    print(f"jz shape: {jz.shape}")
    jx_cut = jx[:, :, SI:EI]
    jy_cut = jy[:, :, SI:EI]
    jz_cut = jz[:, :, SI:EI]
    print(f"jz_cut shape: {jz_cut.shape}")
    cut_off = jz_cut.max()/cutoff
    multiplier = 1/(2*np.sqrt(jx_cut**2 + jy_cut**2 + jz_cut**2).max())

    # Sixth last arg is the divider for the real space grid, multiplier gives a thicker diameter
    print("Writing .spt files..")
    utils_zcolor.plot_current(
        jx,
        jy,
        jz,
        x_cor,
        y_cor,
        z_cor,
        path + "current",
        grid_size,
        multiplier,
        cut_off,
        path,
        align1,
        align2
    )

    utils_zcolor.plot_convergence()


if __name__ == '__main__':
	exit(main())
