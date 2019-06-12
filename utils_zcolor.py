import numpy as np
from ase import Atoms
from ase.io import write
from ase.io import read
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def plot_basis(atoms, phi_xG, folder_name='./basis'):
    """
    r: coefficients of atom-centered basis functions
    atoms: Atoms-object
    """
    for n, phi in enumerate(phi_xG):
        write('%s/%d.cube' % (folder_name, n), atoms, data=phi)

# TODO: Implement the use of einsum. Slightly faster overall compared to np.linalg.norm
# linalg.norm seems to scale linearly with number of positions.
def distance_matrix(pos):
    # Much faster
    dM = np.array([np.linalg.norm(pos[i] - pos, axis=-1) for i in range(pos.shape[0])])

    return dM

def identify_and_align(molecule, alignatom1, alignatom2):
    vacuum = 4

    pos = molecule.get_positions()

    dM = distance_matrix(pos)

    m = np.unravel_index(np.argmax(dM, axis=None), dM.shape)

    # Electrode
    sI = -1
    eI = -3

    po = (molecule[alignatom1].position)
    lo = (molecule[alignatom2].position)

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

def plot_transmission(energy_grid, trans, save_name):
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


def plot_complex_matrix(matrix, save_name):
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
    ret_gf = np.array([retarded_gf(h_ao, s_ao, energy_grid[en],
                                   gamma_left[:, :, en], gamma_right[:, :, en])
                       for en in range(len(energy_grid))])
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


def orb_grad2(phi_xG, i_orb, j_orb, dx, dy, dz):
    psi = phi_xG[i_orb]
    x, y, z = gradientO4(phi_xG[j_orb], dx, dy, dz)

    return psi*x, psi*y, psi*z


def gradientO4(f, *varargs):
    """Calculate the fourth-order-accurate gradient of an N-dimensional scalar function.
    Uses central differences on the interior and first differences on boundaries
    to give the same shape.
    Inputs:
      f -- An N-dimensional array giving samples of a scalar function
      varargs -- 0, 1, or N scalars giving the sample distances in each direction
    Outputs:
      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.
    """
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    
    if n == 0:
        dx = [1.0]*N

    elif n == 1:
        dx = [varargs[0]]*N

    elif n == N:
        dx = list(varargs)

    else:
        raise SyntaxError("invalid number of arguments")

    # use central differences on interior and first differences on endpoints

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)]*N
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = np.zeros(f.shape, f.dtype.char)

        slice0[axis] = slice(2, -2)
        slice1[axis] = slice(None, -4)
        slice2[axis] = slice(1, -3)
        slice3[axis] = slice(3, -1)
        slice4[axis] = slice(4, None)

        # 1D equivalent -- out[2:-2] = (f[:4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
        out[tuple(slice0)] = (f[tuple(slice1)]
                       - 8.0*f[tuple(slice2)]
                       + 8.0*f[tuple(slice3)]
                       - f[tuple(slice4)]
                       ) / 12.0

        slice0[axis] = slice(None, 2)
        slice1[axis] = slice(1, 3)
        slice2[axis] = slice(None, 2)

        # 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        out[tuple(slice0)] = (f[tuple(slice1)] - f[tuple(slice2)])

        slice0[axis] = slice(-2, None)
        slice1[axis] = slice(-2, None)
        slice2[axis] = slice(-3, -1)

        # 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        out[tuple(slice0)] = (f[tuple(slice1)] - f[tuple(slice2)])

        # divide by step size
        outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice(None)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if N == 1:
        return outvals[0]

    else:
        return outvals


def Jc_current(Gles, phi_xg, gd0, path, data_basename, fname):
    Mlt = 1j*Gles/(4*np.pi)
    n = len(Mlt)
    np.save(path + data_basename + "Gles_dV.npy", Mlt)

    jx = gd0.zeros(1)[0]
    jy = gd0.zeros(1)[0]
    jz = gd0.zeros(1)[0]

    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)

    dx = x_cor[1] - x_cor[0]
    dy = y_cor[1] - y_cor[0]
    dz = z_cor[1] - z_cor[0]

    bf_list = np.arange(0, n, 1)
    for k, i in enumerate(bf_list):
        for l, j in enumerate(bf_list):
            x1, y1, z1 = orb_grad2(phi_xg, i, j, dx, dy, dz)

            jx += 2*Mlt[k, l].real*x1
            jy += 2*Mlt[k, l].real*y1
            jz += 2*Mlt[k, l].real*z1

    dA = (x_cor[1] - x_cor[0])*(y_cor[1] - y_cor[0])
    current = jz.sum(axis=(0, 1))*dA

    return current, jx, jy, jz, x_cor, y_cor, z_cor


def create_colorlist(colors):
    n_bins = [201]  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
    colorlist = []

    for n in np.arange(n_bins[0]):
        colorlist.append(list(cm(n))[:-1])

    return colorlist


def plot_current(jx, jy, jz, x, y, z, savename, s, amp, co, path, align1, align2):
    refmol = read(path + 'central_region.xyz')
    refcoord1 = refmol[align1 + 1].position
    refcoord2 = refmol[align2 + 1].position

    # make colorlist for zcolor
    colors = [[140/255., 0, 255/255.], [1, 1, 1], [255/255., 165/255., 0]]  # R -> G -> B
    z_colorlist = create_colorlist(colors)

    # make colorlist for cylindrical color
    colors = [[1, 0, 0], [1, 1, 1], [0, 0, 1]]  # R -> G -> B
    cyl_colorlist = create_colorlist(colors)

    au2A = 0.529177249
    x = x[::s]*au2A
    y = y[::s]*au2A
    z = z[::s]*au2A

    jz = jz[::s, ::s, ::s]
    jy = jy[::s, ::s, ::s]
    jx = jx[::s, ::s, ::s]

    cyl_list = []
    z_list = []

    z_list.append('load "file:$SCRIPT_PATH$/central_region.xyz" \n'.format(path))
    z_list.append('write "$SCRIPT_PATH$/central_region2.xyz" \n'.format(path))
    z_list.append('load "file:$SCRIPT_PATH$/central_region2.xyz" \n'.format(path))

    cyl_list.append('load "file:$SCRIPT_PATH$/central_region.xyz" \n'.format(path))
    cyl_list.append('write "$SCRIPT_PATH$/central_region2.xyz" \n'.format(path))
    cyl_list.append('load "file:$SCRIPT_PATH$/central_region2.xyz" \n'.format(path))

    a = 0
    size = 4 # or 8 for smaller arrows
    
    for ix, x2 in enumerate(x):
        for iy, y2 in enumerate(y):
            for iz, z2 in enumerate(z):
                norm2 = np.sqrt(jx[ix, iy, iz]**2 + jy[ix, iy, iz]**2 + jz[ix, iy, iz]**2)
                norm = np.sqrt(jz[ix, iy, iz]**2)

                if norm2 > co and z2 > refcoord1[2] and z2 < refcoord2[2]:
                    rel_z = jz[ix, iy, iz]/norm2
                    z_color = z_colorlist[int(np.round(rel_z, decimals=2)*100) + 100]

                    rel_phi = (jy[ix, iy, iz]*np.cos(np.arctan2(y2 - refcoord1[1], x2 - refcoord1[0])) - jx[ix, iy, iz]*np.sin(np.arctan2(y2 - refcoord1[1], x2 - refcoord1[0])))/norm2
                    cyl_color = cyl_colorlist[int(np.round(rel_phi, decimals=2)*100) + 100]

                    z_list.append("draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".
                    format(a,
                           x2 - jx[ix, iy, iz]/(size*norm2), y2 - jy[ix, iy, iz]/(size*norm2), z2 - jz[ix, iy, iz]/(size*norm2),
                           (x2 + jx[ix, iy, iz]/(size*norm2)),
                           (y2 + jy[ix, iy, iz]/(size*norm2)),
                           (z2 + jz[ix, iy, iz]/(size*norm2)),
                           norm*amp,
                           z_color))

                    cyl_list.append("draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".
                    format(a,
                           x2 - jx[ix, iy, iz]/(size*norm2), y2 - jy[ix, iy, iz]/(size*norm2), z2 - jz[ix, iy, iz]/(size*norm2),
                           (x2 + jx[ix, iy, iz]/(size*norm2)),
                           (y2 + jy[ix, iy, iz]/(size*norm2)),
                           (z2 + jz[ix, iy, iz]/(size*norm2)),
                           norm*amp,
                           cyl_color))
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
