import numpy as np
from ase import Atoms
from ase.io import write
from ase.io import read
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

c = ["#007fff", "#ff3616"]  # Blue, Red


def get_eigenchannels(Gr_approx, GamL):
    A = np.dot(np.dot(Gr_approx, GamL), Gr_approx.T.conj())
    T_eigs, V_eigs = np.linalg.eig(A)
    order = np.argsort(T_eigs)
    T_eigs = T_eigs.take(order)
    V_eigs = V_eigs.take(order, axis=1)

    return T_eigs, V_eigs


def plot_eigenchannels(
    atoms: Atoms,
    teig_phi_xg,
    folder_name: str = "./basis/eigenchannels"
) -> None:
    for idx, phi in enumerate(teig_phi_xg):
        write(f"{folder_name}/{idx}.cube", atoms, data=phi)


def plot_basis(atoms, phi_xG, folder_name='./basis') -> None:
    """
    atoms: Atoms-object
    """
    for n, phi in enumerate(phi_xG):
        write(f'{folder_name}/{n}.cube', atoms, data=phi)


def identify_and_align(molecule, alignatom1: int, alignatom2: int):
    vacuum = 4

    # Electrode
    sI = -1
    eI = -3

    po = molecule[alignatom1].position
    lo = molecule[alignatom2].position

    v = lo - po
    z = [0, 0, 1]
    molecule.rotate(v, z)

    molecule.center(vacuum=vacuum)

    elec1 = Atoms('H', positions=[molecule[sI].position])
    elec2 = Atoms('H', positions=[molecule[eI].position])

    del molecule[eI]
    del molecule[sI]

    atoms = elec1 + molecule + elec2

    atoms.center(vacuum=vacuum)
    atoms.set_pbc([1, 1, 1])

    return atoms


def plot_transmission(energy_grid, trans, save_name: str) -> None:
    plt.plot(energy_grid, trans)
    plt.yscale('log')
    plt.ylabel(r'Transmission')
    plt.xlabel(r'E-E$_F$ (eV)')
    plt.savefig(save_name)
    plt.close()


def calc_trans(energy_grid, gret, gamma_left, gamma_right):
    """
    Landauer Transmission
    NOTE: using numpy.linalg.multi_dot takes longer than consecutive numpy.dot.
    Presumably because the matrices are pretty much the same and the overhead
    from calculating efficiency outweighs doing the calculation.
    """
    trans = np.array([np.matmul(np.matmul(np.matmul(gamma_left[:, :, en], gret[:, :, en]), gamma_right[:, :, en]), gret[:, :, en].T.conj()).trace() for en in range(len(energy_grid))])

    return trans


def plot_complex_matrix(matrix, save_name: str) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2)

    for dat, axis in zip([matrix.real, matrix.imag], axes.flat):
        image = axis.matshow(dat, cmap='seismic', vmin=dat.min(), vmax=dat.max())
        cb = fig.colorbar(image, ax=axis, orientation='horizontal')
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        plt.cm.get_cmap('seismic')

    plt.savefig(save_name)
    plt.close()


def retarded_gf(h_ao, s_ao, energy, gamma_left, gamma_right):
    """
    Retarded Gf using approx
    """
    return np.linalg.inv(energy*s_ao - h_ao + (1j/2.)*(gamma_left + gamma_right))


def retarded_gf2(h_ao, s_ao, energy, sigma_ret):
    """
    Retarded Gf
    """
    eta = 1e-10

    return np.linalg.inv(-sigma_ret + (energy + eta*1.j)*s_ao - h_ao)


def ret_gf_ongrid(energy_grid, h_ao, s_ao, gamma_left, gamma_right):
    """
    Put the retarded gf on an energy grid
    """
    ret_gf = np.array(
        [retarded_gf(h_ao, s_ao, energy_grid[en], gamma_left[:, :, en], gamma_right[:, :, en]) for en in range(len(energy_grid))]
    )
    ret_gf = np.swapaxes(ret_gf, 0, 2)

    return ret_gf


def fermi(energy, mu_):
    """
    Fermi-Dirac distribution
    """
    # Less computationally robust method.
    # old = 1./(np.exp((energy - mu_)/kbt_) + 1.)
    kbt_ = 25e-6

    return 0.5 + 0.5*np.tanh(-((energy - mu_)/kbt_)/2)


def fermi_ongrid(energy_grid, e_f, bias):
    f_left = []
    f_right = []

    for en in energy_grid:
        f_left.append(fermi(en, e_f + bias/2.))
        f_right.append(fermi(en, e_f - bias/2.))

    return f_left, f_right


def create_colorlist(colors):
    n_bins = [201]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
    colorlist = []

    for n in np.arange(n_bins[0]):
        colorlist.append(list(cm(n))[:-1])

    return colorlist


def plot_current(
    jx,
    jy,
    jz,
    x,
    y,
    z,
    savename: str,
    grid_size,
    multiplier: float,
    cut_off: float,
    path: str,
    align1: int,
    align2: int,
) -> None:
    refmol = read(path + 'central_region.xyz')
    refcoord1 = refmol[align1 + 1].position
    refcoord2 = refmol[align2 + 1].position

    # make colorlist for zcolor
    colors = [[140/255., 0, 255/255.], [1, 1, 1], [255/255., 165/255., 0]]  # Y -> W -> P
    z_colorlist = create_colorlist(colors)

    # make colorlist for cylindrical color
    colors = [[1, 0, 0], [1, 1, 1], [0, 0, 1]]  # R -> W -> B
    cyl_colorlist = create_colorlist(colors)

    au2A = 0.529177249
    x = x[::grid_size]*au2A
    y = y[::grid_size]*au2A
    z = z[::grid_size]*au2A

    jz = jz[::grid_size, ::grid_size, ::grid_size]
    jy = jy[::grid_size, ::grid_size, ::grid_size]
    jx = jx[::grid_size, ::grid_size, ::grid_size]

    cyl_list = []
    z_list = []

    z_list.append('load "file:$SCRIPT_PATH$/central_region.xyz" \n')
    z_list.append('write "$SCRIPT_PATH$/central_region2.xyz" \n')
    z_list.append('load "file:$SCRIPT_PATH$/central_region2.xyz" \n')

    cyl_list.append('load "file:$SCRIPT_PATH$/central_region.xyz" \n')
    cyl_list.append('write "$SCRIPT_PATH$/central_region2.xyz" \n')
    cyl_list.append('load "file:$SCRIPT_PATH$/central_region2.xyz" \n')

    size = 5  # or 8 for smaller arrows

    a = 0
    for ix, x2 in enumerate(x):
        for iy, y2 in enumerate(y):
            for iz, z2 in enumerate(z):
                norm2 = np.sqrt(
                    jx[ix, iy, iz]**2 + jy[ix, iy, iz]**2 + jz[ix, iy, iz]**2
                )
                norm = np.sqrt(jz[ix, iy, iz]**2)

                if norm2 > cut_off and z2 > refcoord1[2] and z2 < refcoord2[2]:
                    rel_z = jz[ix, iy, iz]/norm2
                    z_color = z_colorlist[
                        int(np.round(rel_z, decimals=2)*100) + 100
                    ]

                    rel_phi = (jy[ix, iy, iz]*np.cos(np.arctan2(y2 - refcoord1[1], x2 - refcoord1[0])) - jx[ix, iy, iz]*np.sin(np.arctan2(y2 - refcoord1[1], x2 - refcoord1[0])))/norm2
                    cyl_color = cyl_colorlist[int(np.round(rel_phi, decimals=2)*100) + 100]

                    z_list.append("draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".format(
                        a,
                        x2 - jx[ix, iy, iz]/(size*norm2),
                        y2 - jy[ix, iy, iz]/(size*norm2),
                        z2 - jz[ix, iy, iz]/(size*norm2),
                        x2 + jx[ix, iy, iz]/(size*norm2),
                        y2 + jy[ix, iy, iz]/(size*norm2),
                        z2 + jz[ix, iy, iz]/(size*norm2),
                        norm*multiplier,
                        z_color)
                    )

                    cyl_list.append("draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".format(
                        a,
                        x2 - jx[ix, iy, iz]/(size*norm2),
                        y2 - jy[ix, iy, iz]/(size*norm2),
                        z2 - jz[ix, iy, iz]/(size*norm2),
                        x2 + jx[ix, iy, iz]/(size*norm2),
                        y2 + jy[ix, iy, iz]/(size*norm2),
                        z2 + jz[ix, iy, iz]/(size*norm2),
                        norm*multiplier,
                        cyl_color)
                    )
                a += 1

    z_list.append("set defaultdrawarrowscale 0.1 \n")
    z_list.append('rotate 90 \n')
    z_list.append('background white \n')

    cyl_list.append("set defaultdrawarrowscale 0.1 \n")
    cyl_list.append('rotate 90 \n')
    cyl_list.append('background white \n')

    with open(savename + "_zcolor.spt", "w") as text_file:
        text_file.writelines(z_list)

    with open(savename + "_cylcolor.spt", "w") as text_file:
        text_file.writelines(cyl_list)


def plot_convergence() -> None:
    _, _, jz, x, y, z = np.load(
        "./data/data/current_all.npy", allow_pickle=True
    )
    current = np.abs(
        np.real(np.load("./data/data/current_dV_mo.npy", allow_pickle=True))
    )*6.623618183e-3
    z *= 0.529177249*0.1
    dA = (x[1] - x[0])*(y[1] - y[0])
    jz_sum = np.abs(jz.sum(axis=(0, 1))*dA*6.623618183e-3)

    fig, ax = plt.subplots()
    ax.plot(z, jz_sum, c=c[0], label="xy-plane current")
    ax.axhline(current, linestyle="--", c=c[1], label="Total current")
    print(f"Current on convergence plot: {current}")

    ax.set_xlabel("z-coordinate (nm)")
    ax.set_ylabel("Current (A)")
    
    plt.legend(frameon=False)
    plt.savefig("./data/convergence_plot.png", dpi=150)
