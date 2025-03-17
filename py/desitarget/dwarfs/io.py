"""
desitarget.streams.io
=====================

Reading/writing data for the MWS Dwarf Galaxy programs.

Largely a wrapper of existing functions from desitarget.streams.io
"""
import os
import numpy as np
from desitarget import io
from desitarget.streams.io import read_data_per_stream_one_file, \
    read_data_per_stream
from desiutil import depend

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM the Gaia Data Release for matching throughout this module.
gaiadr = "dr3"


def read_data_per_dwarf_one_file(filename, ra, dec, maxd,
                                 mindec=-20., readall=False):
    """Assemble the data needed for a dwarf program from one file.
    Wrapper for desitarget.streams.io.read_data_per_stream_one_file()

    Parameters
    ----------
    swdir : :class:`str`
        Name of a Legacy Surveys sweep file.
    ra, dec : :class:`float`
        Central coordinates of dwarf galaxy in DEGREES.
    maxd : :class:`float` or `int`
        Maximum angular distance from the center of the dwarf
        coordinate system to search for members in DEGREES.
    mindec : :class:`float` or `int`, optional, defaults to -20 (20oS)
        Hard limit on data (objects south of this are not returned).
    readall : :class:`bool`, optional, defaults to ``False``
        Ignore the stream-related inputs (`decpol`, `mind`, `maxd`) and
        instead read _all_ of the sweep files.

    Returns
    -------
    :class:`array_like`
        An array of objects from the filename that are in the dwarf,
        with matched Gaia information.
    """
    return read_data_per_stream_one_file(filename, ra, dec, 0, maxd)


def read_data_per_dwarf(swdir, ra, dec, maxd, dwarf_name,
                        readcache=True, addnors=True, test=False, numproc=1,
                        mindec=-20, readall=False):
    """Assemble the data needed for a particular dwarf program.
    Wrapper for desitarget.streams.io.read_data_per_stream()

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    ra, dec : :class:`float`
        Central coordinates of dwarf galaxy in DEGREES.
    maxd : :class:`float` or `int`
        Maximum angular distance from the center of the dwarf
        coordinate system to search for members in DEGREES.
    dwarf_name : :class:`str`
        Name of a dwarf. Used to make the cached filename, e.g. "DRACO_1".
    readcache : :class:`bool`, optional, defaults to ``True``
        If ``True`` read from a previously constructed and cached file
        automatically, IF such a file exists. If ``False`` don't read
        from the cache AND OVERWRITE the cached file, if it exists. The
        cached file is $TARG_DIR/streamcache/dwarfname-drX-cache.fits,
        where dwarfname is the lower-case passed `dwarf_name` and drX
        is the Legacy Surveys Data Release (parsed from `swdir`).
    addnors : :class:`bool`, optional, defaults to ``True``
        If ``True`` then if `swdir` contains "north" add sweep files from
        the south by substituting "south" in place of "north" (and vice
        versa, i.e. if `swdir` contains "south" add sweep files from the
        north by substituting "north" in place of "south").
    test : :class:`bool`, optional, defaults to ``False``
        Read a subset of the data for testing purposes.
    numproc : :class:`int`, optional, defaults to 1 for serial
        The number of parallel processes to use. `numproc` of 16 is a
        good balance between speed and file I/O.
    mindec : :class:`float` or `int`, optional, defaults to -20 (20oS)
        Hard limit on data (objects south of this are not returned).
    readall : :class:`bool`, optional, defaults to ``False``
        Ignore all of the other inputs except for `addnors` and instead
        read (and cache) _all_ of the sweep files.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.

    Notes
    -----
    - Example values for, e.g., DRACO_1:
        swdir = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0"
        ra, dec = 260.0684,  57.9185
        maxd = 4
    - The $TARG_DIR environment variable must be set to read/write from
      a cache. If $TARG_DIR is not set, caching is completely ignored.
    - This is useful for a single dwarf.

    """
    return read_data_per_stream(swdir, ra, dec, 0, maxd, dwarf_name,
                                readcache=readcache, addnors=addnors, test=test, 
                                numproc=numproc)


def write_targets(dirname, targs, header, dwarfnames="", obscon=None,
                  subpriority=True):
    """Write dwarf targets to a FITS file.
    Functionally the same as desitarget.streams.io.write_targets()
    except streamtargets -> dwarftargets

    Parameters
    ----------
    dirname : :class:`str`
        The output directory name. Filenames are constructed from other
        inputs.
    targs : :class:`~numpy.ndarray`
        The numpy structured array of data to write.
    header : :class:`dict`
        Header for output file. Can be a FITShdr object or dictionary.
        Pass {} if you have no additional header information.
    dwarfnames : :class:`str, optional
        Information about dwarf names that correspond to the targets.
        Included in the output filename.
    obscon : :class:`str`, optional, defaults to `None`
        Can pass one of "DARK" or "BRIGHT". If passed, don't write the
        full set of data, rather only write targets appropriate for
        "DARK" or "BRIGHT" observing conditions. The relevant
        `PRIORITY_INIT` and `NUMOBS_INIT` columns will be derived from
        `PRIORITY_INIT_DARK`, etc. and `filename` will have "bright" or
        "dark" appended to the lowest DIRECTORY in the input `filename`.
   subpriority : :class:`bool`, optional, defaults to ``True``
        If ``True`` and a `SUBPRIORITY` column is in the input `targs`,
        then `SUBPRIORITY==0.0` entries are overwritten by a random float
        in the range 0 to 1, using a seed of 816.

    Returns
    -------
    :class:`int`
        The number of targets that were written to file.
    :class:`str`
        The name of the file to which targets were written.

    Notes
    -----
    - Must contain at least the columns:
        PHOT_G_MEAN_MAG, PHOT_BP_MEAN_MAG, PHOT_RP_MEAN_MAG and
        FIBERTOTFLUX_G, FIBERTOTFLUX_R, FIBERTOTFLUX_Z, RELEASE
    - Always OVERWRITES existing files!
    - Writes atomically. Any output files that died mid-write will be
      appended by ".tmp".
    - Units are automatically added from the desitarget units yaml file
      (see `/data/units.yaml`).
    - Mostly wraps :func:`~desitarget.io.write_with_units`.
    """
    # ADM limit to just BRIGHT or DARK targets, if requested.
    # ADM Ignore the filename output, we'll build that on-the-fly.
    if obscon is not None:
        _, header, targs = io._bright_or_dark(dirname, header, targs, obscon)

    # ADM construct the output filename.
    drs = list(set(targs["RELEASE"]//1000))
    if len(drs) == 1:
        drint = drs[0]
        drstr = f"dr{drint}"
    else:
        log.info("Couldn't parse LS data release. Defaulting to drX.")
        drint = "X"
        drstr = "drX"
    outfn = f"dwarftargets-{dwarfnames.lower()}-bright.fits"
    outfn = os.path.join(dirname, drstr, io.desitarget_version,
                         "dwarftargets", "main", "resolve", "bright", outfn)

    # ADM check if any targets are too bright.
    maglim = 15
    fluxlim = 10**((22.5-maglim)/2.5)
    toobright = np.zeros(len(targs), dtype="?")
    for col in ["GAIA_PHOT_G_MEAN_MAG", "GAIA_PHOT_BP_MEAN_MAG",
                "GAIA_PHOT_RP_MEAN_MAG"]:
        toobright |= (targs[col] != 0) & (targs[col] < maglim)
    for col in ["FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z"]:
        toobright |= (targs[col] != 0) & (targs[col] > fluxlim)
    if np.any(toobright):
        tids = targs["TARGETID"][toobright]
        log.warning(f"Targets TOO BRIGHT to be written to {outfn}: {tids}")
        # ADM remove the targets that are too bright.
        targs = targs[~toobright]

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in targs.dtype.names and subpriority:
        subpseed = 816
        np.random.seed(subpseed)
        # SB only set subpriorities that aren't already set, but keep
        # original full random sequence order.
        ii = targs["SUBPRIORITY"] == 0.0
        targs["SUBPRIORITY"][ii] = np.random.random(len(targs))[ii]
        header["SUBPSEED"] = subpseed

    # ADM add the DESI dependencies.
    depend.add_dependencies(header)
    # ADM some other useful header information.
    depend.setdep(header, 'desitarget', io.desitarget_version)
    depend.setdep(header, 'desitarget-git', io.gitversion())
    depend.setdep(header, 'photcat', drstr)

    # ADM add information to construct the filename to the header.
    header["OBSCON"] = "bright"
    header["SURVEY"] = "main"
    header["RESOLVE"] = True
    header["DR"] = drint
    header["GAIADR"] = gaiadr

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    # ADM and, finally, write out the targets.
    io.write_with_units(outfn, targs, extname="DWARFTARGETS", header=header)

    return len(targs), outfn
