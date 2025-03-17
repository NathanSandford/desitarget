import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from desitarget.dwarfs.utilities import get_dwarf_parameters, apply_plx_zpt, get_plx_error, transform2center


def visualize_selection(D, sel1, sel2, sel3, sel4, sel5, sel6, dwarf_name, input_cat=None, lit_mem=None, tile_centers=None):
    """Visualize target selection.

    Parameters
    ----------

    Returns
    -------
    None
    """
    # look up the defining parameters of the dwarf.
    dwarf = get_dwarf_parameters(dwarf_name)
    # the parameters that define the coordinates of the dwarf.
    ra0, dec0 = dwarf["RA"], dwarf["DEC"]
    # the parameters that define the proper motion of the dwarf.
    pmra0, pmdec0 = dwarf["PMRA"], dwarf["PMDEC"]
    # the parameters that define the morphology of the dwarf.
    rhalf = dwarf["RHALF"]
    ellipticity = dwarf["ELLIPTICITY"]
    position_angle = dwarf["POSITION_ANGLE"]
    # distance to the dwarf
    dist = dwarf["DIST"]
    # retrieve the color and magnitude offsets.
    coloff = dwarf["COLOFF"]
    magoff = dwarf["MAGOFF"]
    # the isochrone of the dwarf
    iso_rgb_g = np.array(dwarf["ISO_RGB_G"])
    iso_rgb_r = np.array(dwarf["ISO_RGB_R"])
    iso_hb_g = np.array(dwarf["ISO_HB_G"])
    iso_hb_r = np.array(dwarf["ISO_HB_R"])
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
    g0 = g - eg
    r0 = r - er
    z0 = z - ez
    g0_r0 = g0 - r0
    r0_z0 = r0 - z0
    Gmag = D['PHOT_G_MEAN_MAG']
    RPmag = D['PHOT_RP_MEAN_MAG']
    G_RP = Gmag - RPmag
    # Calculate dRA, dDec
    dRA, dDec = transform2center(D['RA'], D['DEC'], ra0, dec0)
    # Calculate distance modulus
    dm = 5 * np.log10(dist * 1e3) - 5
    # Apply zero-point correction to parallax
    plx = apply_plx_zpt(D)
    # Get parallax error
    parallax_error = get_plx_error(D)

    # Select All Targets
    sel0 = (sel1 | sel2 | sel3 | sel4 | sel5 | sel6)
    
    fig = plt.figure(figsize=(21, 31))
    gs = GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    
    # spatial plot
    ax1.scatter(
        dRA[sel6],
        dDec[sel6],
        marker='.', alpha=0.1, c='pink',
        label='FILLER'
    )
    ax1.scatter(
        dRA[sel5],
        dDec[sel5],
        marker='.', alpha=0.1, c='k',
        label='FAINT_NO_PM'
    )
    ax1.scatter(
        dRA[sel4],
        dDec[sel4],
        marker='.', alpha=0.5, c='purple',
        label='PM_ONLY'
    )
    ax1.scatter(
        dRA[sel3],
        dDec[sel3],
        marker='.', alpha=0.5, c='y',
        label='BRIGHT_PM3'
    )
    ax1.scatter(
        dRA[sel2],
        dDec[sel2],
        marker='.', alpha=0.5, c='orange',
        label='BRIGHT_PM2'
    )
    ax1.scatter(
        dRA[sel1],
        dDec[sel1],
        marker='.', alpha=0.5, c='r',
        label='BRIGHT_PM1'
    )
    ax1.scatter(0, 0, marker='*', s=500, c='y', ec='k', zorder=1000)
    elliptical_radii = [1, 10, 20]
    for eradius in elliptical_radii:
        ell = Ellipse(
            xy=(0, 0),
            width=2 * eradius * rhalf,
            height=2 * eradius * rhalf * (1 - ellipticity),
            angle=90 - position_angle,
            zorder=30,
            fc=None,
            ec="k",
            lw=4,
            fill=False,
            alpha=0.5,
        )
        ax1.add_artist(ell)
        ax1.text(
            eradius * rhalf * np.cos(np.deg2rad(90 - position_angle)),
            eradius * rhalf * np.sin(np.deg2rad(90 - position_angle)),
            f'{eradius}' + r'$r_h$',
            color='k',
            fontsize=24,
            zorder=20,
        )
    if tile_centers is not None:
        tile_dRA, tile_dDec = transform2center(
            tile_centers[0], tile_centers[1], ra0, dec0
        )
        for i in range(len(tile_dRA)):
            ell = Ellipse(
                xy=(tile_dRA[i], tile_dDec[i]),
                width=2 * 1.6,
                height=2 * 1.6,
                fc=None,
                #fc="white",
                ec="white",
                lw=4,
                ls='-',
                fill=False,
                #fill=True,
                alpha=0.8,
                #alpha=0.1,
                zorder=100,
            )
            ax1.add_artist(ell)
    ax1.set_xlabel(r"$\Delta$ RA", fontsize=24)
    ax1.set_ylabel(r"$\Delta$ DEC", fontsize=24)
    ax1.tick_params(labelsize=16)

    # proper motion plot
    ax2.scatter(
        D['PMRA'][sel6],
        D['PMDEC'][sel6],
        marker='.', alpha=0.1, c='pink'
    )
    ax2.scatter(
        D['PMRA'][sel4],
        D['PMDEC'][sel4],
        marker='.', alpha=0.5, c='purple'
    )
    ax2.scatter(
        D['PMRA'][sel3],
        D['PMDEC'][sel3],
        marker='.', alpha=0.5, c='y'
    )
    ax2.scatter(
        D['PMRA'][sel2],
        D['PMDEC'][sel2],
        marker='.', alpha=0.5, c='orange'
    )
    ax2.scatter(
        D['PMRA'][sel1],
        D['PMDEC'][sel1],
        marker='.', alpha=0.5, c='r'
    )
    ax2.scatter(pmra0, pmdec0, marker='*', s=500, c='y', ec='k', zorder=1000)
    ax2.set_xlabel("PMRA", fontsize=24)
    ax2.set_ylabel("PMDEC", fontsize=24)
    ax2.tick_params(labelsize=16)
    
    # DECaLS CMD plot
    ax3.scatter(
        g0_r0[sel6], 
        r0[sel6], 
        marker='.', alpha=0.1, c='pink'
    )
    ax3.scatter(
        g0_r0[sel5], 
        r0[sel5], 
        marker='.', alpha=0.1, c='k'
    )
    ax3.scatter(
        g0_r0[sel4], 
        r0[sel4], 
        marker='.', alpha=0.5, c='purple'
    )
    ax3.scatter(
        g0_r0[sel3], 
        r0[sel3], 
        marker='.', alpha=0.5, c='y'
    )
    ax3.scatter(
        g0_r0[sel2], 
        r0[sel2], 
        marker='.', alpha=0.5, c='orange'
    )
    ax3.scatter(
        g0_r0[sel1], 
        r0[sel1], 
        marker='.', alpha=0.5, c='r'
    )
    ax3.plot(iso_rgb_gr, iso_rgb_r + dm, c='y', lw=4)
    ax3.plot(iso_hb_gr, iso_hb_r + dm, c='y', lw=4)
    ax3.set_xlabel("g0-r0", fontsize=24)
    ax3.set_ylabel("r0", fontsize=24)
    ax3.tick_params(labelsize=16)
    
    # Gaia CMD plot
    ax4.scatter(
        G_RP[sel6], 
        RPmag[sel6], 
        marker='.', alpha=0.1, c='pink'
    )
    ax4.scatter(
        G_RP[sel4], 
        RPmag[sel4], 
        marker='.', alpha=0.5, c='purple'
    )
    ax4.scatter(
        G_RP[sel3], 
        RPmag[sel3], 
        marker='.', alpha=0.5, c='y'
    )
    ax4.scatter(
        G_RP[sel2], 
        RPmag[sel2], 
        marker='.', alpha=0.5, c='orange'
    )
    ax4.scatter(
        G_RP[sel1], 
        RPmag[sel1], 
        marker='.', alpha=0.5, c='r'
    )
    ax4.set_xlabel("G-RP", fontsize=24)
    ax4.set_ylabel("RP", fontsize=24)
    ax4.tick_params(labelsize=16)
    
    ax5.hist(
        (plx / parallax_error)[sel6],
        histtype='step',
        color='pink',
        bins=np.linspace(-10, 10, 100),
    )
    ax5.hist(
        (plx / parallax_error)[sel4],
        histtype='step',
        color='purple',
        bins=np.linspace(-10, 10, 100),
    )
    ax5.hist(
        (plx / parallax_error)[sel3],
        histtype='step',
        color='y',
        bins=np.linspace(-10, 10, 100),
    )
    ax5.hist(
        (plx / parallax_error)[sel2],
        histtype='step',
        color='orange',
        bins=np.linspace(-10, 10, 100),
    )
    ax5.hist(
        (plx / parallax_error)[sel1],
        histtype='step',
        color='r',
        bins=np.linspace(-10, 10, 100),
    )
    ax5.set_xlabel("Parallax / Parallax Error", fontsize=24)
    ax5.set_ylabel("# Stars", fontsize=24)
    ax5.set_yscale('log')
    ax5.tick_params(labelsize=16)
    
    if lit_mem is not None:
        # Calculate dRA, dDec
        dRA, dDec = transform2center(lit_mem['RA'], lit_mem['DEC'], ra0, dec0)
        ax1.scatter(
            dRA,
            dDec,
            marker='x', alpha=0.3, c='b',
            label='Lit. Mem.'
        )
        ax2.scatter(
            lit_mem['PMRA'], 
            lit_mem['PMDEC'], 
            marker='x', alpha=0.3, c='b'
        )
        ax3.scatter(
            lit_mem['g0'] - lit_mem['r0'], 
            lit_mem['r0'], 
            marker='x', alpha=0.3, c='b'
        )
        ax4.scatter(
            lit_mem['PHOT_G_MEAN_MAG'] - lit_mem['PHOT_RP_MEAN_MAG'], 
            lit_mem['PHOT_RP_MEAN_MAG'], 
            marker='x', alpha=0.3, c='b'
        )
        # Apply zero-point correction to parallax
        plx = apply_plx_zpt(lit_mem)
        # Get parallax error
        parallax_error = get_plx_error(lit_mem)
        ax5.hist(
            plx / parallax_error,
            histtype='step',
            color='b',
            bins=np.linspace(-10, 10, 100),
        )
    if input_cat is not None:
        # dust correction.
        ext_coeff = dict(g=3.237, r=2.176, z=1.217)
        eg, er, ez = [ext_coeff[_] * input_cat['EBV'] for _ in 'grz']
        g, r, z = [22.5 - 2.5 * np.log10(input_cat['FLUX_' + _]) for _ in 'GRZ']
        g0 = g - eg
        r0 = r - er
        z0 = z - ez
        g0_r0 = g0 - r0
        r0_z0 = r0 - z0
        Gmag = input_cat['PHOT_G_MEAN_MAG']
        RPmag = input_cat['PHOT_RP_MEAN_MAG']
        G_RP = Gmag - RPmag
        # Calculate dRA, dDec
        dRA, dDec = transform2center(input_cat['RA'], input_cat['DEC'], ra0, dec0)
        # Calculate distance modulus
        dm = 5 * np.log10(dist * 1e3) - 5
        # Apply zero-point correction to parallax
        plx = apply_plx_zpt(D)
        # Get parallax error
        parallax_error = get_plx_error(D)
        
        ax1.hist2d(dRA, dDec, bins=(500, 500), cmap='gray', norm='log', zorder=0)
        ax2.hist2d(
            input_cat['PMRA'][np.isfinite(input_cat['PMRA'])], 
            input_cat['PMDEC'][np.isfinite(input_cat['PMRA'])], 
            bins=(1000, 1000), cmap='gray', norm='log', zorder=0
        )
        ax3.hist2d(
            g0_r0[np.isfinite(g0_r0)], r0[np.isfinite(g0_r0)], 
            bins=(1000, 500), cmap='gray', norm='log', zorder=0
        )
        ax4.hist2d(
            G_RP[np.isfinite(G_RP) & np.isfinite(RPmag)], 
            RPmag[np.isfinite(G_RP) & np.isfinite(RPmag)], 
            bins=(1000, 500), cmap='gray', norm='log', zorder=0
        )
        ax5.hist(
            (plx / parallax_error),
            histtype='step',
            color='k',
            bins=np.linspace(-10, 10, 100),
        )
    ax2.set_xlim(
        np.nanquantile(D['PMRA'][sel0], 0.01) - 5,
        np.nanquantile(D['PMRA'][sel0], 0.99) + 5,
    )
    ax2.set_ylim(
        np.nanquantile(D['PMDEC'][sel0], 0.01) - 5,
        np.nanquantile(D['PMDEC'][sel0], 0.99) + 5,
    )
    ax3.set_ylim(22.5, 15.0)
    ax3.set_xlim(-0.5, 1.5)
    ax4.set_ylim(22, 14)
    ax4.set_xlim(-0.5, 1.5)
    ax5.set_xlim(-10, 10)
    ax1.legend(fontsize=16, loc='upper left')
    return fig
