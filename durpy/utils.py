import os, math, io, time as _t, contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio
from rasterio.transform import from_bounds
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_hex
from matplotlib.patches import Patch

def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")[:48]

def _mcmap(p):
    cm = LinearSegmentedColormap.from_list("c", p) if p else plt.get_cmap("viridis")
    cm = cm.copy(); cm.set_bad((1,1,1,0)); return cm

def _listed(cols):
    lc = ListedColormap(cols); lc.set_bad((1,1,1,0)); return lc

#############################################################################
def _sample_colors(palette, k):
    cm = LinearSegmentedColormap.from_list("c", palette)
    return [to_hex(cm(x)) for x in np.linspace(0.12, 0.92, k)]

###############################################################################
def _fmt(v):
    """Formatage numérique adaptatif pour les légendes de classes."""
    a = abs(v)
    if a == 0:    return "0"
    if a >= 100:  return "%.0f" % v
    if a >= 10:   return "%.1f" % v
    if a >= 1:    return "%.2f" % v
    if a >= 0.1:  return "%.3f" % v
    return "%.4g" % v

#################################################################################
def _decor(ax, ext):
    """Graticule, barre d'échelle et flèche Nord — identiques au notebook."""
    ax.grid(True, ls="-", lw=0.3, color="0.8", alpha=0.6, zorder=4)
    ax.tick_params(top=True, right=True, labelsize=8)
    kmdeg = 111.32 * math.cos(math.radians((ext[2] + ext[3]) / 2.0))
    tgt   = (ext[1] - ext[0]) * kmdeg * 0.25
    e     = 10 ** math.floor(math.log10(max(tgt, 1e-6)))
    nice  = e
    for mlt in (1, 2, 5, 10):
        if mlt * e >= tgt: nice = mlt * e; break
    bar = nice / kmdeg
    x0  = ext[0] + (ext[1] - ext[0]) * 0.06
    y0  = ext[2] + (ext[3] - ext[2]) * 0.05
    ax.plot([x0, x0+bar], [y0, y0], color="k", lw=3, solid_capstyle="butt", zorder=7)
    ax.text(x0 + bar/2, y0 + (ext[3]-ext[2])*0.012, "%g km" % nice,
            ha="center", va="bottom", fontsize=8, zorder=7)
    ax.annotate("N", xy=(0.96, 0.96), xytext=(0.96, 0.88),
                xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold",
                arrowprops=dict(facecolor="k", width=3, headwidth=9))