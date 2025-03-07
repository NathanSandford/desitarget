"""
desitarget.dwarfs.utilties
===========================

Utilities for the DESI MWS Dwarf Galaxy programs.

Adapted from desitarget.streams.utilities,
which borrows heavily from Sergey Koposov's `astrolibpy routines`_.

Log of changes from desitarget.streams.utilities:
- get_dwarf_parameters() replaces get_stream_parameters()
- cmd_sel_func() replaces get_CMD_interpolator()
- spatial_sel_func() added
- get_plx_error() and apply_plx_zpt() split from plx_sel_func()
- plx_sel_func() edited to allow for negative parallax
- Non-dwarf specific functions removed, including:
    - ivars_to_errors()
    - errors_to_ivars()
    - cosd()
    - sind()
    - betw()
- Unneeded transforms removed, including:
    - torect()
    - fromrect()
    - rotation_matrix()
    - sphere_rotate()
    - rotate_pm()
    - correct_pm()
    

.. _`astrolibpy routines`: https://github.com/segasai/astrolibpy/blob/master/my_utils
"""

import yaml
import numpy as np
import astropy.coordinates as acoo
import astropy.units as auni
from importlib import resources
from scipy.optimize import curve_fit
import desitarget.streams.gaia_dr3_parallax_zero_point.zpt as gaia_zpt
from desitarget.streams.utilities import betw

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def get_dwarf_parameters(dwarf_name):
    """Look up information for a given dwarf galaxy.
    Functionally the same as desitarget.streams.utilities.get_stream_parameters()
    except stream_name -> dwarf_name and streams.yaml -> dwarfs.yaml

    Parameters
    ----------
    dwarf_name : :class:`str`
        Name of a dwarf galaxy that appears in the ../data/dwarfs.yaml file.
        Possibilities include 'DRACO_1'.

    Returns
    -------
    :class:`~dict`
        A dictionary of dwarf galaxy parameters for the passed `dwarf_name`.
        Includes isochrones and positional information.

    Notes
    -----
    - Parameters for each stream are in the ../data/dwarfs.yaml file.
    """
    # guard against name being passed as lower-case.
    dwarf_name = dwarf_name.upper()

    # open and load the parameter yaml file.
    fn = resources.files('desitarget').joinpath('data/dwarfs.yaml')
    with open(fn) as f:
        dwarf = yaml.safe_load(f)

    return dwarf[dwarf_name]


def spatial_sel_func(ra0, dec0, maxd, D):
    """Selects dwarf members within a given radius.
    Currently redundant with the pre-selection used in generating the input catalog.

    Parameters
    ----------
    ra0, dec0 : :class:`float`
        Central coordinates of dwarf galaxy in DEGREES.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA` and `DEC`.
    maxd : :class:`~numpy.ndarray` or `float`
        Maximum distance of possible targets to dwarf.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # coordinates of the dwarf.
    cdwarf = acoo.SkyCoord(ra0 * auni.degree, dec0 * auni.degree)
    cobjs = acoo.SkyCoord(D['RA'] * auni.degree, D['DEC'] * auni.degree)
    # separation between the objects of interest and the dwarf.
    sep = cobjs.separation(cdwarf)
    # lies in the the dwarf
    return sep.value < maxd


def pm_sel_func(pmra0, pmdec0, D, pad=2, mult=2.5):
    """Select dwarf members using proper motion, padded by some error.

    Parameters
    ----------
    pmra0, pmdec0 : :class:`~numpy.ndarray` or `float`
        Systemic proper motions of dwarf galaxy.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `PMRA`, `PMDEC`, `PMRA_ERROR`, and `PMDEC_ERROR`.
    pad: : :class:`float` or `int`, defaults to 2
        Extra offset with which to pad `mult`*proper_motion_error.
    mult : :class:`float` or `int`, defaults to 2.5
        Multiple of the proper motion error to use for padding.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # combine proper motion errors in RA and Dec
    pm_err = np.sqrt(0.5 * (D['PMRA_ERROR']**2 + D['PMDEC_ERROR']**2))
    return np.sqrt((D['PMRA'] - pmra0)**2 + (D['PMDEC'] - pmdec0)**2) < pad + mult * pm_err


def get_plx_error(D):
    """Gets parallax error from inverse variance if necessary.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains either
        the column `PARALLAX_ERROR` or the column `PARALLAX_IVAR`.
    Returns
    -------
    :class:`array_like` or `float`
        Parallax error.
    """
    if 'PARALLAX_ERROR' in D.dtype.names:
        parallax_error = D['PARALLAX_ERROR']
    elif 'PARALLAX_IVAR' in D.dtype.names:
        parallax_error = np.zeros_like(D["PARALLAX_IVAR"]) + 1e8
        ii = lit_mem['PARALLAX_IVAR'] != 0
        parallax_error[ii] = 1./np.sqrt(D[ii]['PARALLAX_IVAR'])
    else:
        msg = "Either PARALLAX_ERROR or PARALLAX_IVAR must be passed!"
        log.error(msg)
    return parallax_error


def apply_plx_zpt(D):
    """Select dwarf members using parallax, padded by some error.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA`, `ASTROMETRIC_PARAMS_SOLVED`, `PHOT_G_MEAN_MAG`,
        `NU_EFF_USED_IN_ASTRONOMY`, `PSEUDOCOLOUR`, `ECL_LAT`, `PARALLAX`
        `PARALLAX_ERROR`. `PARALLAX_IVAR` will be used instead of
        `PARALLAX_ERROR` if `PARALLAX_ERROR` is not present.

    Returns
    -------
    :class:`array_like` or `float`
        Zero-point corrected parallax.
    """
    subset = np.in1d(D['ASTROMETRIC_PARAMS_SOLVED'], [31, 95])
    plx_zpt_tmp = gaia_zpt.get_zpt(D['PHOT_G_MEAN_MAG'][subset],
                                   D['NU_EFF_USED_IN_ASTROMETRY'][subset],
                                   D['PSEUDOCOLOUR'][subset],
                                   D['ECL_LAT'][subset],
                                   D['ASTROMETRIC_PARAMS_SOLVED'][subset],
                                   _warnings=False)
    plx_zpt = np.zeros(len(D['RA']))
    plx_zpt_tmp[~np.isfinite(plx_zpt_tmp)] = 0
    plx_zpt[subset] = plx_zpt_tmp
    plx = D['PARALLAX'] - plx_zpt
    return plx


def plx_sel_func(dist, D, plx_sys=0.05, mult=2.5, keep_all_neg=False, min_plx_plxerr=-5):
    """Select dwarf members using parallax, padded by some error.
    NOTE: This is a slight alteration on the GD1 parallax selection,
    as it will select anything with a negative parallax 
    regardless of the distance to the galaxy

    Parameters
    ----------
    dist : :class:`~numpy.ndarray` or `float`
        Distance of possible stream members.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA`, `ASTROMETRIC_PARAMS_SOLVED`, `PHOT_G_MEAN_MAG`,
        `NU_EFF_USED_IN_ASTRONOMY`, `PSEUDOCOLOUR`, `ECL_LAT`, `PARALLAX`
        `PARALLAX_ERROR`. `PARALLAX_IVAR` will be used instead of
        `PARALLAX_ERROR` if `PARALLAX_ERROR` is not present.
    plx_sys : :class:`float`
        Extra offset with which to pad `mult`*parallax_error.
    mult : :class:`float` or `int`
        Multiple of the parallax error to use for padding.
    keep_all_neg : :class:`bool`
        If true, keep all targets with negative parallaxes.
    min_plx_plxerr : :class:`float`
        Minimum allowable parallax/parallax error, defaults to -5.
        Useful if keep_all_neg = True.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """
    # Apply zero-point correction to parallax
    plx = apply_plx_zpt(D)
    # Get parallax error
    parallax_error = get_plx_error(D)
    # Apply systemic error padding
    dplx = plx - 1 / dist
    if keep_all_neg:
        sel = (dplx < plx_sys + mult * parallax_error) \
            & (plx / parallax_error > min_plx_plxerr)
    else:
        sel = (np.abs(dplx) < plx_sys + mult * parallax_error) \
            & (plx / parallax_error > min_plx_plxerr)
    return sel


def cmd_sel_func(
    dwarf_name, D, 
    rgb_color_tol=0.2, rgb_mag_tol=0.0, 
    rgb_grmin=-0.5, rgb_grmax=1.5,
    rgb_rmin=16, rgb_rmax=23,
    hb_color_tol=0.1, hb_mag_tol=0.5,
    hb_grmin=-0.5, hb_grmax=0.5, 
    hb_rmin=16, hb_rmax=23,
    show_magerr_plot=False
):
    """Select dwarf galaxy members using CMD cuts.

    Parameters
    ----------
    dwarf_name : :class:`str`
        Name of a dwarf galaxy that appears in the ../data/dwarfs.yaml file.
        Possibilities include 'DRACO_1'.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `FLUX_G`, `FLUX_R`, `FLUX_Z`,
        `FLUX_IVAR_G`, `FLUX_IVAR_R`, `FLUX_IVAR_Z`,
        and `EBV`.
    rgb_color_tol : :class:`float` or `int`
        Width of RGB cmd selection, defaults to 0.2.
    rgb_mag_tol : :class:`float` or `int`
        Height of RGB cmd selection, defaults to 0.0.
        WARNING: Non-zero values lead to unexpected behavior at the tip of the RGB
    rgb_grmin : :class:`float` or `int`
        Minimum extinction-corrected g-r for RGB selection, defaults to -0.5.
    rgb_grmax : :class:`float` or `int`
        Maximum extinction-corrected g-r for RGB selection, defaults to 1.5.
    rgb_rmin : :class:`float` or `int`
        Bright limit for RGB selection, defaults to 16.
    rgb_rmax : :class:`float` or `int`
        Faint limit for RGB selection, defaults to 23.
    hb_color_tol : :class:`float` or `int`
        Width of HB cmd selection, defaults to 0.1.
    hb_mag_tol : :class:`float` or `int`
        Height of HB cmd selection, defaults to 0.5.
    hb_grmin : :class:`float` or `int`
        Minimum extinction-corrected g-r for HB selection, defaults to -0.5.
    hb_grmax : :class:`float` or `int`
        Maximum extinction-corrected g-r for HB selection, defaults to 0.5.
    hb_rmin : :class:`float` or `int`
        Bright limit for HB selection, defaults to 16.
    hb_rmax : :class:`float` or `int`
        Faint limit for HB selection, defaults to 23.
    show_magerr_plot : :class:`bool`
        Show log10(rmag  error) vs. rmag relationship

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # look up the defining parameters of the dwarf.
    dwarf = get_dwarf_parameters(dwarf_name)
    # distance to the dwarf
    dist = dwarf["DIST"]
    # distance modulus of the dwarf
    dm = 5 * np.log10(dist * 1e3) - 5
    # the RGB isochrone of the dwarf
    iso_rgb_g = np.array(dwarf["ISO_RGB_G"])
    iso_rgb_r = np.array(dwarf["ISO_RGB_R"])
    # the HB isochrone of the dwarf
    iso_hb_g = np.array(dwarf["ISO_HB_G"])
    iso_hb_r = np.array(dwarf["ISO_HB_R"])
    # retrieve the color and magnitude offsets.
    coloff = dwarf["COLOFF"]
    magoff = dwarf["MAGOFF"]
    
    # apply isochrone offsets
    iso_rgb_gr = iso_rgb_g - iso_rgb_r
    iso_hb_gr = iso_hb_g - iso_hb_r
    iso_rgb_r += magoff
    iso_hb_r += magoff
    iso_rgb_gr -= coloff
    iso_hb_gr -= coloff

    # dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * D['EBV'] for _ in 'grz']
    g, r, z = [22.5 - 2.5 * np.log10(D['FLUX_' + _]) for _ in 'GRZ']
    gerr, rerr, zerr = [2.5 / np.log(10) * (np.sqrt(1./D['FLUX_IVAR_'+_]) / D['FLUX_' + _]) for _ in 'GRZ']
    g0 = g - eg
    r0 = r - er
    z0 = z - ez

    # fit log10(rmag error) vs rmag relation
    def log10_error_func(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(
        log10_error_func, 
        r[betw(r, 15, 24) & betw(np.log10(rerr), -4, 0)], 
        np.log10(rerr)[betw(r, 15, 24) & betw(np.log10(rerr), -4, 0)]
    )
    if show_magerr_plot:
        plt.figure(figsize=(5, 5))
        plt.scatter(
            r[betw(r, 15, 24) & betw(np.log10(rerr), -4, 0)],
            np.log10(rerr)[betw(r, 15, 24) & betw(np.log10(rerr), -4, 0)],
            marker='.',
            alpha=0.1,
            c='k'
        )
        xdata = np.linspace(15, 24, 100)
        plt.plot(xdata, log10_error_func(xdata, *popt))
        plt.xlabel("rmag")
        plt.ylabel("log10(rmagerr)")
        plt.show()

    # magnitude cut along RGB
    mag_sel_rgb = betw(r0, np.min(iso_rgb_r + dm) - 0.5, rgb_rmax) & betw(g0 - r0, rgb_grmin, rgb_grmax)
    # color cut along RGB
    rgb_color_tol = np.sqrt(rgb_color_tol**2 + (3*10**log10_error_func(iso_rgb_r + dm, *popt))**2)
    grmax1 = np.interp(r0, iso_rgb_r[::-1] + dm, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    grmax2 = np.interp(r0, iso_rgb_r[::-1] + dm + rgb_mag_tol, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    grmax3 = np.interp(r0, iso_rgb_r[::-1] + dm - rgb_mag_tol, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    rgb_grmax_cmd = np.max(np.array([grmax1, grmax2, grmax3]), axis=0)
    grmin1 = np.interp(r0, iso_rgb_r[::-1] + dm, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    grmin2 = np.interp(r0, iso_rgb_r[::-1] + dm - rgb_mag_tol, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    grmin3 = np.interp(r0, iso_rgb_r[::-1] + dm + rgb_mag_tol, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    rgb_grmin_cmd = np.min(np.array([grmin1, grmin2, grmin3]), axis=0)
    color_sel_rgb = betw(g0 - r0, rgb_grmin_cmd, rgb_grmax_cmd)
    # final RGB selection
    rgb_sel = mag_sel_rgb & color_sel_rgb

    # magnitude cut along HB
    mag_sel_hb = betw(r0, hb_rmin, hb_rmax) & betw(g0 - r0, hb_grmin, hb_grmax)
    # color cut along HB
    gr_hb = np.interp(r0, iso_hb_r[::-1] + dm , iso_hb_gr[::-1], left=np.nan, right=np.nan)
    rr_hb = np.interp(g0 - r0, iso_hb_gr, iso_hb_r + dm, left=np.nan, right=np.nan)
    color_sel_hb = mag_sel_hb & ((abs((g0 - r0) - gr_hb) < hb_color_tol) | (abs(r0 - rr_hb) < hb_mag_tol))
    # final HB selection
    hb_sel = mag_sel_hb & color_sel_hb

    return rgb_sel | hb_sel
