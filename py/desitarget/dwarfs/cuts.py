"""
desitarget.streams.cuts
=======================

Target selection cuts for the DESI MWS dSph programs.

Borrows heavily from Sergey Koposov and the DESI stream target selection cuts.
"""
from time import time
import numpy as np
from desitarget.streams.utilities import betw
from desitarget.dwarfs.utilities import get_dwarf_parameters, \
    spatial_sel_func, pm_sel_func, plx_sel_func, cmd_sel_func
from desitarget.dwarfs.io import read_data_per_dwarf
from desitarget.cuts import _psflike
from desitarget.targets import resolve
from desitarget.streams.targets import finalize

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# CMR modified for DESI extension
def is_in_dwarf(objs, dwarf_name):
    """Performs target selection on a source catalog for a given dwarf galaxy.

    Parameters
    ----------
    objs : :class:`array_like`
        Numpy rec array with at least the Legacy Surveys/Gaia columns:
        RA, DEC, PARALLAX, PMRA, PMDEC, PARALLAX_IVAR, PMRA_IVAR,
        PMDEC_IVAR, EBV, FLUX_G, FLUX_R, FLUX_Z, PSEUDOCOLOUR, TYPE,
        ASTROMETRIC_PARAMS_SOLVED, NU_EFF_USED_IN_ASTROMETRY,
        ECL_LAT, PHOT_G_MEAN_MAG.
    dwarf_name : :class:`str`
        Name of a dwarf galaxy that appears in the ../data/dwarfs.yaml file.
        Possibilities include 'BOOTES_1', 'CANES_VENATICI_1', 'DRACO_1', 'SEXTANS_1', and 'URSA_MINOR_1'.
    visually_validate : :class:`bool`
        Plot spatial, proper motion, and CMD selections


    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "PM_ONLY" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_NO_PM" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # ADM start the clock.
    start = time()
    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # NRS look up the defining parameters of the dwarf.
    dwarf = get_dwarf_parameters(dwarf_name)
    # NRS galaxy coordinates in degrees.
    ra0, dec0 = dwarf["RA"], dwarf["DEC"]
    # NRS galaxy proper motions in mas/yr.
    pmra0, pmdec0 = dwarf["PMRA"], dwarf["PMDEC"]
    # NRS spatial extent in degrees for initial data read.
    maxd = dwarf["MAXD"]
    # NRS galaxy distance in kpc.
    dist = dwarf["DIST"]

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    g, r, z = [22.5 - 2.5 * np.log10(objs['FLUX_' + _]) for _ in 'GRZ']
    eg, er, ez = [ext_coeff[_] * objs['EBV'] for _ in 'grz']
    gerr, rerr, zerr = [2.5 / np.log(10) * (np.sqrt(1./objs['FLUX_IVAR_'+_]) / objs['FLUX_' + _]) for _ in 'GRZ']
    g0 = g - eg
    r0 = r - er
    z0 = z - ez
    g0_r0 = g0 - r0
    r0_z0 = r0 - z0

    # NRS spatial selection; currently redundant with catalog creation
    field_sel = spatial_sel_func(ra0, dec0, maxd, objs)
    log.info(f"Objects in the field: {field_sel.sum()}...t={time()-start:.1f}s")

    # NRS Gaia-based selection (proper motion, parallax, bright limit).
    gaia_pm_sel = pm_sel_func(
        pmra0, pmdec0, objs, pad=dwarf['PM_PAD'], mult=dwarf['PM_NSIG']
    )
    gaia_plx_sel = plx_sel_func(
        dist, objs, plx_sys=dwarf['PLX_SYS'], mult=dwarf['PLX_NSIG'],
        keep_all_neg=True, min_plx_plxerr=-5
    )
    gaia_astrom_sel = gaia_pm_sel & gaia_plx_sel
    log.info(f"With correct astrometry: {(gaia_astrom_sel & field_sel).sum()}")

    # NRS CMD selection
    cmd_sel = cmd_sel_func(dwarf_name, objs)

    # NRS magnitude ranges
    brightpm1_magsel = (r > dwarf['BRIGHT_LIMIT']) & (z <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = betw(z, dwarf['BRIGHTPM1_LIMIT'], dwarf['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = betw(z, dwarf['BRIGHTPM2_LIMIT'], dwarf['BRIGHTPM3_LIMIT'])
    pm_only_magsel = (r > dwarf['BRIGHT_LIMIT']) & (z <= stream['PM_ONLY_LIMIT'])
    faint_no_pm_magsel = betw(z, dwarf['FAINT_NO_PM_LIMIT'], dwarf['FAINT_LIMIT'])
    filler_magsel = betw(z, dwarf['FILLER_LIMIT'], dwarf['FAINT_LIMIT'])

    # NRS FILLER stellar locus selection.
    stellar_locus_blue_sel = (
        betw(r0_z0 - (-0.17 + 0.67 * g0_r0), -0.2, 0.2)
        & (g0_r0 <= 1.1)
    )
    stellar_locus_red_sel = (
        betw(g0_r0 - (1.05 + 0.25 * r0_z0), -0.2, 0.2)
        & (g0_r0 > 1.1)
    )
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel   

    # NRS BRIGHT_PM targets
    bright_pm1 = cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3
    log.info(f"Objects meeting BRIGHT_PM selection: {bright_pm.sum()}, " 
             + f"(by bin: {bright_pm1.sum()}/{bright_pm2.sum()}/{bright_pm3.sum()})" 
             + f"...t={time()-start:.1f}s")

    # NRS PM_ONLY targets
    pm_only = ~cmd_sel & gaia_astrom_sel & field_sel & pm_only_magsel & betw(g0_r0, -0.3, 1.3)
    log.info(f"Objects meeting PM_ONLY selection: {pm_only.sum()}...t={time()-start:.1f}s")

    # NRS FAINT_NO_PM targets
    faint_no_pm = cmd_sel & field_sel & faint_no_pm_magsel & ~np.isfinite(objs['PMRA']) & _psflike(objs["TYPE"])
    log.info(f"Objects meeting FAINT_NO_PM selection: {faint_no_pm.sum()}...t={time()-start:.1f}s")

    # NRS FILLER targets
    filler = field_sel & filler_magsel & stellar_locus_sel & betw(g0_r0, -0.3, 1.2) & _psflike(objs["TYPE"]) & ~bright_pm & ~pm_only & ~faint_no_pm

    # CMR moved these here so we write numbers of the final selections, but less useful for timing
    log.info(f"Objects meeting BRIGHTPM selection: {np.sum(bright_pm)}")
    log.info(f"Objects meeting BRIGHTPM1 selection: {np.sum(bright_pm1)}")
    log.info(f"Objects meeting BRIGHTPM2 selection: {np.sum(bright_pm2)}")
    log.info(f"Objects meeting BRIGHTPM3 selection: {np.sum(bright_pm3)}")
    log.info(f"Objects meeting FAINT_NO_PM selection: {np.sum(faint_no_pm)}")
    log.info(f"Objects meeting PM_ONLY selection: {np.sum(faint_no_pm)}")
    log.info(f"Objects meeting FILLER selection: {np.sum(filler)}")
    log.info(f"Finished selection for {dwarf_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + pm_only.astype(int) + faint_no_pm.astype(int) + filler.astype(int)
    if np.max(check) > 1:
        msg = "Selections should be unique but they overlap!"
        log.error(msg)

    return bright_pm1, bright_pm2, bright_pm3, pm_only, faint_no_pm, filler



def set_target_bits(objs, dwarf_names=['BOOTES_1', 'CANES_VENATICI_1', 'DRACO_1', 'SEXTANS_1', 'URSA_MINOR_1']):
    """Select dwarf targets, returning target mask arrays.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns needed for
        dwarf target selection. See, e.g.,
        :func:`~desitarget.dwarf.cuts.is_in_dwarf` for column names.
    dwarf_names : :class:`list`
        A list of dwarf galaxy names to process. Defaults to all dwarfs.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target, scnd_target) where each
        element is an array of target selection bitmasks for each object.

    Notes
    -----
    - See ../data/targetmask.yaml for the definition of each bit.
    """
    from desitarget.targetmask import desi_mask, mws_mask

    # ADM set up a zerod mws_target array to |= with later.
    # CMR changed to mws
    mws_target = np.zeros_like(objs["RA"], dtype='int64')

    # ADM might be able to make this more general by putting the
    # ADM bit names in the data/yaml file and using globals()
    # ADM to recover the is_in() functions.

    # NRS loop over all dwarfs in extension
    for dwarf_name in dwarf_names:
        bright_pm1, bright_pm2, bright_pm3, pm_only, faint_no_pm, filler = is_in_dwarf(
            objs, dwarf_name
        )
        # CMR set mws desi extension bit
        mws_target |= (mws_target != 0) * mws_mask.MWS_EXT
        # NRS set dwarf name bit
        mws_target |= (mws_target != 0) * mws_mask[f"MWS_{dwarf_name}"]
        # CMR set target subclass bit masks
        mws_target |= bright_pm1 * mws_mask.MWS_BRIGHT_PM1
        mws_target |= bright_pm2 * mws_mask.MWS_BRIGHT_PM2
        mws_target |= bright_pm3 * mws_mask.MWS_BRIGHT_PM3
        mws_target |= pm_only * mws_mask.MWS_PM_ONLY
        mws_target |= faint_no_pm * mws_mask.MWS_FAINT_NO_PM
        mws_target |= filler * mws_mask.MWS_FILLER

    # ADM tell DESI_TARGET where MWS_ANY was updated.
    # CMR updated to MWS 
    desi_target = (mws_target != 0) * desi_mask.MWS_ANY

    # OBSOLETE: ADM set BGS_TARGET and MWS_TARGET to zeros.
    # CMR guessed scnd_target needs to get set to zero now
    bgs_target = np.zeros_like(mws_target)
    scnd_target = np.zeros_like(mws_target)

    return desi_target, bgs_target, mws_target, scnd_target


def select_targets(
    swdir, 
    dwarf_names=['BOOTES_1', 'CANES_VENATICI_1', 'DRACO_1', 'SEXTANS_1', 'URSA_MINOR_1'], 
    readperdwarf=True,
    addnors=True, 
    readcache=True,
):
    """Process files from an input directory to select targets.

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    dwarf_names : :class:`list`
        A list of dwarf galaxy names to process. Defaults to all dwarfs.
    readperdwarf : :class:`bool`, optional, defaults to ``True``
        When set, read each dwarf's data individually instead of looping
        through all possible sweeps files. This is likely quickest and
        most useful when working with a single dwarf. For multiple
        dwarfs it may cause issues when duplicate targets are selected.
    addnors : :class:`bool`
        If ``True`` then if `swdir` contains "north" add sweep files from
        the south by substituting "south" in place of "north" (and vice
        versa, i.e. if `swdir` contains "south" add sweep files from the
        north by substituting "north" in place of "south").
    readcache : :class:`bool`, optional, defaults to ``True``
        If ``True`` read all data from previously made cache files,
        in cases where such files exist. If ``False`` don't read
        from caches AND OVERWRITE any cached files, if they exist. Cache
        files are named $TARG_DIR/streamcache/dwarfname-drX-cache.fits,
        where dwarfname is the lower-case name from `dwarf_names` and
        drX is the Legacy Surveys Data Release (parsed from `swdir`).

    Returns
    -------
    :class:`~numpy.ndarray`
        Targets in the input `swdir` which pass the cuts with added
        targeting columns such as ``TARGETID``, and ``DESI_TARGET``,
        ``BGS_TARGET``, ``MWS_TARGET``, ``SCND_TARGET`` (i.e. target
        selection bitmasks).
    """
    if readperstream:
        # ADM loop over dwarfs and read in the data per-dwarf.
        # ADM eventually, for multiple dwarfs, we would likely switch
        # ADM to read in each sweep file and parallelizing across files.
        allobjs = []
        for dwarf_name in dwarf_names:
            # NRS look up the defining parameters of the dwarf.
            dwarf = get_dwarf_parameters(dwarf_name)
            # NRS galaxy coordinates in degrees.
            ra0, dec0 = dwarf["RA"], dwarf["DEC"]
            # NRS spatial extent in degrees for initial data read.
            maxd = dwarf["MAXD"]
            # NRS read in the data.
            objs = read_data_per_dwarf(swdir, ra0, dec0, maxd, dwarf_name,
                                        addnors=addnors, readcache=readcache)
            allobjs.append(objs)
        objects = np.concatenate(allobjs)
    else:
        # ADM --TODO-- write loop across sweeps instead of streams.
        msg = ("readperdwarf must be True until we implement looping "
               "over sweeps instead of dwarfs")
        log.error(msg)

    # ADM process the targets.
    desi_target, bgs_target, mws_target, scnd_target = set_target_bits(
        objects, dwarf_names=dwarf_names)

    # ADM finalize the targets.
    # ADM anything with DESI_TARGET !=0 is truly a target.
    ii = (desi_target != 0)
    objects = objects[ii]
    desi_target = desi_target[ii]
    bgs_target = bgs_target[ii]
    mws_target = mws_target[ii]
    scnd_target = scnd_target[ii]

    # ADM add TARGETID and targeting bitmask columns.
    targets = finalize(objects, desi_target, bgs_target, mws_target, scnd_target)

    # ADM resolve any duplicates between imaging data releases.
    targets = resolve(targets)

    # ADM we'll definitely need to update the read_data loop if we ever
    # ADM have overlapping targets in overlapping streams/dwarfs.
    if len(np.unique(targets["TARGETID"])) != len(targets):
        msg = ("Targets are not unique. The code needs updated to read in the "
               "sweep files one-by-one (as in desitarget.cuts.select_targets()) "
               "rather than caching each individual dwarf")
        log.error(msg)

    return targets
