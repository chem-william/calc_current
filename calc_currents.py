# This code was tested using Python 3.6.4, GPAW 1.5.1, and ASE 3.16.0


# This script must be run with access to utils_zcolor.py and with an appropriately setup 'data' directory which can be generated with the make_dirs.sh shell script.
# The output is a jmol (.spt) script, and can be collected with the collect_spt.py script.

# For help contact marc@chem.ku.dk / marchamiltongarner@hotmail.com

# Please cite the appropriate work.
# s-band electrode transmission code: DOI: 10.1021/acs.jpclett.8b03432
# Current density code: Jensen et al. "When Current Does Not Follow Bonds: Current Density In Saturated Molecules" Submitted 2018
# Current density with cylindrical coordinates: Garner et al. "Helical Orbitals and Circular Currents in Linear Carbon Wires" Submitted 2018

import argparse
import os
import sys
# To retrieve path to utils_zcolor which should be placed in the same folder as
# calc_current.py
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/bin/py_scripts/')
import pickle

from numpy import ascontiguousarray as asc
from gpaw import GPAW
from gpaw import FermiDirac
from gpaw.lcao.tools import dump_hamiltonian_parallel
from gpaw.lcao.tools import get_bfi2
from ase import Atoms
from ase.io import read
from ase.io import write
from ase.units import Hartree
import numpy as np
import matplotlib
matplotlib.use('agg')

import utils_zcolor


def read_config(config_file):
	with open(config_file) as file:
		lines = file.readlines()
	return int(lines[0].strip()), int(lines[1].strip())

def main():	
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--path',
			    default='data/',
			    help='path to data folder')
	parser.add_argument('--xyzname',
			    default='hh_junc.traj',
			    help='name of the xyz or traj file')
	parser.add_argument('--basis',
	                    default='dzp',
	                    help='basis (sz, dzp, ...)')
	parser.add_argument('--ef',
	                    default=0.,
	                    help='fermi')
	parser.add_argument('--config',
			    default=None,
			    help='name of the config file where start and end indices are')
	args = parser.parse_args()
	path = os.path.abspath(args.path) + "/"
	ef = float(args.ef)
	basis = args.basis
	xyzname = args.xyzname
	config = args.config


	# Constants
	xc = 'PBE'
	FDwidth = 0.1
	kpts = (1, 1, 1)
	mode = 'lcao'
	h = 0.20
	vacuum = 4

	eV2au = 1/Hartree

	co = 1e-10 * eV2au
	bias = 1e-3 * eV2au
	gamma = 1e0 * eV2au
	ef = ef * eV2au # If ef has been set to a custom value.
	estart, eend = [-6 * eV2au, 6 * eV2au]
	es = 1e-2 * eV2au
	energy_grid = np.arange(estart, eend, es)

	basis_full = {'H': 'sz',
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
	              'Ru': basis}

	# Divider for h - size of the arrows for current density
	grid_size = 3

	basename = "__basis_{0}__h_{1}__cutoff_{2}__xc_{3}__gridsize_{4:.2f}__bias_{5}__ef_{6}__gamma_{7}__energy_grid_{8}_{9}_{10}__multi_grid__type__".format(
	    basis,
	    h,
	    co,
	    xc,
	    grid_size,
	    bias / eV2au,
	    ef / eV2au,
	    gamma / eV2au,
	    estart / eV2au,
	    eend / eV2au,
	    es / eV2au)

	fname = "basis_{0}__xc_{1}__h_{2}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(
	    basis,
	    xc,
	    h,
	    FDwidth,
	    kpts,
	    mode,
	    vacuum)

	plot_basename = "plots/" + basename
	data_basename = "data/" + basename

	molecule = read(path + xyzname)

	# Align z-axis and cutoff at these atoms, OBS paa retningen.
	if config == None:
		raise ValueError('No config file has been chosen')
	align1, align2 = read_config(config)

	# Identify end atoms and align according to z-direction
	atoms = utils_zcolor.identify_and_align(molecule, align1, align2)
	symbols = atoms.get_chemical_symbols()
	np.save(path + "positions.npy", atoms.get_positions())
	np.save(path + "symbols.npy", symbols)
	atoms.write(path + "central_region.xyz")

	# Run and converge calculation
	calc = GPAW(h=h,
	            xc=xc,
	            basis=basis_full,
	            occupations=FermiDirac(width=FDwidth),
	            kpts=kpts,
	            mode=mode,
	            symmetry={'point_group': False, 'time_reversal': False},
	            charge=0,
	            txt='logfile.txt')

	atoms.set_calculator(calc)
	atoms.get_potential_energy()  # Converge everything!
	print('fermi is', atoms.calc.get_fermi_level())

	dump_hamiltonian_parallel(path + 'scat_' + fname, atoms, direction='z')

	# Write AO basis to disk
	bfs = get_bfi2(symbols, basis, range(len(atoms)))
	rot_mat = np.diag(v=np.ones(len(bfs)))
	c_fo_xi = asc(rot_mat.real.T)  # coefficients
	phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
	gd0 = calc.wfs.gd

	calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, phi_xg, -1)
	# Writing this to disk while going multiprocessing with GPAW
	# Reason: Numpy tries to pickle the objects, but gd0 is 
	# not pickleable as it's a MPI object
	# np.save(path + fname + "ao_basis_grid", [phi_xg, gd0])
	utils_zcolor.plot_basis(atoms, phi_xg, folder_name=path + "basis/ao")

	# Calculate the transmission AJ style - AO basis
	print("Calculating transmission - AO-basis")

	H_ao, S_ao = pickle.load(open(path + 'scat_' + fname + '0.pckl', 'rb'))
	H_ao = H_ao[0, 0] * eV2au
	S_ao = S_ao[0]
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
	
	# To optimize the following matrix operations
	Gamma_L = Gamma_L.astype(dtype='float32', order='F')
	Gamma_R = Gamma_R.astype(dtype='float32', order='F')
	
	Gr = Gr.astype(dtype='complex64', order='F')
	trans = utils_zcolor.calc_trans(energy_grid, Gr, Gamma_L, Gamma_R)

	utils_zcolor.plot_transmission(energy_grid*Hartree, np.real(trans), path + plot_basename + "trans.png")
	np.save(path + data_basename + 'trans_full.npy', [energy_grid*Hartree, trans])
	print("AO-transmission done!")

	# Calculate transmission with MO basis
	# Convert AO's to MO's
	eig_vals, eig_vec = np.linalg.eig(np.dot(np.linalg.inv(S_ao), H_ao))
	order = np.argsort(eig_vals)
	eig_vals = eig_vals.take(order)
	eig_vec = eig_vec.take(order, axis=1)
	S_mo = np.dot(np.dot(eig_vec.T.conj(), S_ao), eig_vec)
	H_mo = np.dot(np.dot(eig_vec.T, H_ao), eig_vec)
	eig_vec = eig_vec / np.sqrt(np.diag(S_mo))

	# Save the MO basis as .cube files
	c_fo_xi = asc(eig_vec.real.T)  # coefficients
	mo_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
	calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, mo_phi_xg, -1)
	np.save(path + fname + "mo_energies", eig_vals)
	np.save(path + fname + "mo_basis", mo_phi_xg)
	utils_zcolor.plot_basis(atoms, mo_phi_xg, folder_name=path + "basis/mo")
	# Uncomment to calculate the transmission
	# in MO-basis. If not the same as AO-basis,
	# something is wrong.
	#print("Calculating transmission - MO-basis")

	#GamL_mo = np.dot(np.dot(eig_vec.T, GamL), eig_vec)
	#GamR_mo = np.dot(np.dot(eig_vec.T, GamR), eig_vec)

	#Gamma_L_mo = [GamL_mo for en in range(len(energy_grid))]
	#Gamma_R_mo = [GamR_mo for en in range(len(energy_grid))]
	#Gamma_L_mo = np.swapaxes(Gamma_L_mo, 0, 2)
	#Gamma_R_mo = np.swapaxes(Gamma_R_mo, 0, 2)

	#Gr_mo = utils_zcolor.ret_gf_ongrid(energy_grid, H_mo, S_mo, Gamma_L_mo, Gamma_R_mo)
	#trans_mo = utils_zcolor.calc_trans(energy_grid, Gr_mo, Gamma_L_mo, Gamma_R_mo)
	#utils_zcolor.plot_transmission(energy_grid, np.real(trans_mo), path + plot_basename + "trans_mo.png")
	#np.save(path + data_basename + 'trans_full_mo.npy', [energy_grid, trans_mo])

	#print('MO-transmission done!')
	np.savetxt(path + 'eig_spectrum.txt', X=eig_vals*Hartree, fmt=, newline=\n)
	# find HOMO and LUMO
	for n in range(len(eig_vals)):
	    if eig_vals[n] < 0 and eig_vals[n + 1] > 0:
	        HOMO = eig_vals[n]
	        LUMO = eig_vals[n + 1]
	        midgap = (HOMO + LUMO)/2.0

	        np.savetxt(path + "basis/mo/" + 'homo_index.txt', X=['HOMO index is ', n], fmt='%.10s', newline='\n')
	        break

	hl_gap = ['HOMO is ', HOMO*Hartree, 'LUMO is ', LUMO*Hartree, 'mid-gap is ', midgap*Hartree]
	np.savetxt(path + 'HOMO_LUMO.txt', X=hl_gap, fmt='%.10s', newline='\n')


	"""Current with fermi functions"""
	fR, fL = utils_zcolor.fermi_ongrid(energy_grid, ef, bias)
	dE = energy_grid[1] - energy_grid[0]
	current_trans = (1 / (2*np.pi)) * np.array([trans[en].real * (fL[en]-fR[en])*dE for en in range(len(energy_grid))]).sum()
	np.save(path + data_basename + "current_trans.npy", current_trans)


	"""Current approx at low temp"""
	Sigma_r = -1j/2. * (GamL + GamR)  # + V_pot
	Gr_approx = utils_zcolor.retarded_gf2(H_ao, S_ao, ef, Sigma_r)

	Gles = np.dot(np.dot(Gr_approx, GamL), Gr_approx.T.conj())
	Gles *= bias

	# Sigma_r_mo = -1j/2. * (GamL_mo + GamR_mo)
	# Gr_approx_mo = utils_zcolor.retarded_gf2(H_mo, S_mo, ef, Sigma_r_mo)
	# Gles_mo = np.dot(np.dot(Gr_approx_mo, GamL_mo), Gr_approx_mo.T.conj())

	utils_zcolor.plot_complex_matrix(Gles, path + "Gles")

	Tt = np.dot(np.dot(np.dot(GamL, Gr_approx), GamR), Gr_approx.T.conj())
	# Tt_mo = np.dot(np.dot(np.dot(GamL_mo, Gr_approx_mo), GamR_mo), Gr_approx_mo.T.conj())

	current_dV = (bias/(2*np.pi))*Tt.trace()

	np.save(path + data_basename + "matrices_dV.npy", [Gr_approx, Gles, GamL])
	# np.save(path + data_basename + "matrices_mo_dV.npy", [Gr_approx_mo, Gles_mo, GamL_mo])
	np.save(path + data_basename + "trans_dV.npy", [ef, Tt.trace()])
	# np.save(path + data_basename + "trans_mo_dV.npy", [ef, Tt_mo.trace()])
	np.save(path + data_basename + "current_dV.npy", current_dV)

	"""Non corrected current"""
	current_c, jx_c, jy_c, jz_c, x_cor, y_cor, z_cor = utils_zcolor.Jc_current(Gles,
										   phi_xg,
										   gd0,
										   path,
	 									   data_basename, 
										   fname)

	np.save(path + data_basename + "current_c_all.npy", np.array([jx_c, jy_c, jz_c, x_cor, y_cor, z_cor]))
	np.save(path + data_basename + "current_c.npy", np.array([current_c, x_cor, y_cor, z_cor]))

	SI = 31
	EI = -31
	j_z_cut = jz_c[:, :, SI:EI]
	multiplier = 1/(3*j_z_cut[::2, ::2, ::2].max())
	cut_off = j_z_cut[::2, ::2, ::2].max()/20.

	# Sixth last arg is the divider for the real space grid, multiplier gives a thicker diameter
	utils_zcolor.plot_current(jx_c,
				  jy_c,
				  jz_c,
				  x_cor,
				  y_cor,
				  z_cor,
				  path + "current",
				  grid_size,
				  multiplier,
				  cut_off,
				  path,
				  align1,
				  align2)

if __name__ == '__main__':
	exit(main())
