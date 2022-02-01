# calc_current
Calculates the Landauer transmission and current density for a given molecule in a junction. The electrodes are assumed to be two dihydrogens.

It uses a Rust script to calculate the gradient. The source code can be grabbed [here](https://github.com/chem-william/gradient04)

## How do I use this?
First, a junction has to be generated using `junction.py`. It will look for `.xyz` file in the same folder as itself.
The script attaches two dihydrogens at specified locations at the molecule in the `.xyz`. It creates a `hh_junc.traj` which will subsequently be used for calculating the current density.

To modify where the electrodes are attached to the molecule, you'll have to edit `start_indices`, `end_indices` and the `A`, `D1`, and `D2` parameters of `create_junction()`. The first electrode is attached at a dihedral angle of `D1` formed from the three indices of `start_indices` and `D1`. Likewise for the second electrode, `end_indices`, and `D2`.

The following assumes you're running the calculation on a system that uses [SLURM](https://slurm.schedmd.com/documentation.html).
You need two things in the same folder: a file named hh_junc.traj that contains the junction and a file named config that, as a minimum, contains the indices for the top and bottom atoms on the z-axis. You then run the following command
```bash
current_submit <queue> <CPUs> <mem in GB> <optionally a job name>
```

### Options for the config file
As a minimum the current density calculation requires a top and bottom atom to be specified. These define the z-axis of the molecule and is used to color the arrows by their z-component and φ-component.

Minimal config file
```bash
top_atom=index_of_top_atom
bottom_atom=index_of_bottom_atom
```

You can use [ASE](https://wiki.fysik.dtu.dk/ase/) with the following command to view your molecule and the indices of each atom:
```bash
ase gui molecule.xyz
```
ASE can also show .traj files so you can plot the junction with the following command
```bash
ase gui hh_junc.traj
```

Other options for the config file include the following (the default value is shown):
* ```ef=0``` The energy at which to calculate the current density. The default is the midpoint between the energy of the HOMO and the LUMO
* ```functional=pbe``` The functional that is used to calculate the electronic structure.
* ```h_spacing=0.2``` Specifies the grid spacing in Å that has to be used for the realspace representation of the smooth wave functions
* ```charge=0``` Charge of the molecule
* ```cutoff=20``` How many arrows to include in the plot of the current density. A value of 20 corresponds to not plotting arrows smaller than 5% of the largest arrow.

## Contact
Please feel free to open an issue if you have any questions or notice that something might be wrong!

Otherwise, my contact information can be found [here](https://chem.ku.dk/ansatte/alle/?pure=en%2Fpersons%2Fwilliam-brojoergensen(0abe1d7c-55a6-4da3-b351-57568c14316b).html)
