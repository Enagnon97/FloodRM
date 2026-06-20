import os, math, io, time as _t, contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_hex
from matplotlib.patches import Patch
#from pysheds.grid import Grid

import durpy.variables as _vars
from durpy.variables import LAYERS, _PYSHEDS

def init_local_engine(dem_path):
    """
    Charge le DEM local et initialise les globaux _DEM_ARRAY, _DEM_EXT
    et LAYERS['elevation'] dans durpy.variables.
    """
    import durpy.variables as _vars
    arr, ext = load_raster(dem_path)
    _vars._DEM_ARRAY = arr
    _vars._DEM_EXT   = ext
    _vars.LAYERS["elevation"] = arr
    print(f"DEM chargé : {arr.shape} | ext={ext}")

def fetch_array_local(layer_key=None, arr=None, ext=None):
    import durpy.variables as _vars
    if arr is not None and ext is not None:
        return arr, ext
    if layer_key is not None and layer_key in _vars.LAYERS:
        data = _vars.LAYERS[layer_key]
        if isinstance(data, np.ndarray):
            return data.astype("float32"), _vars._DEM_EXT
    if _vars._DEM_ARRAY is not None:
        return _vars._DEM_ARRAY.copy(), _vars._DEM_EXT
    raise RuntimeError("Aucun DEM chargé. Appelez init_local_engine(dem_path='...') d'abord.")

def load_raster(path):
    """
    Charge n'importe quel GeoTIFF local et retourne (array float32, ext).
    Utilisez cette fonction pour charger des couches dérivées (pente, TWI…).
    """
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            a[a == nd] = np.nan
        b = src.bounds
        ext = [b.left, b.right, b.bottom, b.top]
    return a, ext

def save_raster(arr, ext, path, nodata=-9999.0, crs="EPSG:4326"):
    """
    Enregistre un tableau numpy en GeoTIFF (EPSG:4326 par défaut).
    Crée le dossier parent si nécessaire.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = arr.shape
    transform = from_bounds(ext[0], ext[2], ext[1], ext[3], w, h)
    data = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    with rasterio.open(path, "w", driver="GTiff",
                       height=h, width=w, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data, 1)
    print(f"🗺️  raster enregistré : {path}")

def _ensure_pysheds():
    try:
        from pysheds.grid import Grid
        return Grid
    except ImportError:
        import subprocess, sys
        print("Installation de pysheds (une seule fois, ~30 s)…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pysheds"])
        from pysheds.grid import Grid
        return Grid

def _get_pysheds_grid(dem_arr=None, ext=None):
    """
    Lance pysheds sur le DEM local (fill → flowdir → accumulation).
    Résultat mis en cache dans _PYSHEDS.
    """
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT

    cache_key = id(_vars._DEM_ARRAY)
    if cache_key in _PYSHEDS:
        return _PYSHEDS[cache_key]

    Grid = _ensure_pysheds()

    tmp_path = os.path.join(_vars.TABLE_DIR, "_tmp_dem_pysheds.tif")
    save_raster(dem_arr, ext, tmp_path)

    grid  = Grid.from_raster(tmp_path)
    dem_r = grid.read_raster(tmp_path)
    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem_r)))
    fdir = grid.flowdir(inflated)
    acc  = grid.accumulation(fdir)

    result = dict(grid=grid, fdir=fdir, acc=acc, dem=inflated, path=tmp_path)
    _PYSHEDS[cache_key] = result
    return result

def _ext_of(raster):
    """Retourne [left, right, bottom, top] depuis l'affine d'un raster pysheds."""
    aff = raster.affine; h, w = raster.shape
    return [aff.c, aff.c + aff.a * w, aff.f + aff.e * h, aff.f]
# =============================================================================
# DÉRIVATION DES COUCHES TERRAIN À PARTIR DU DEM LOCAL
# =============================================================================

def derive_slope(dem_arr=None, ext=None, save=True):
    """Calcule la pente (degrés) depuis le DEM local. Retourne (array, ext)."""
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT
    h, w = dem_arr.shape
    dx_deg = (ext[1] - ext[0]) / w   # degrés par pixel
    dy_deg = (ext[3] - ext[2]) / h
    # conversion degrés → mètres (approx)
    lat_c   = (ext[2] + ext[3]) / 2.0
    dx_m    = dx_deg * 111320.0 * math.cos(math.radians(lat_c))
    dy_m    = dy_deg * 111320.0
    gy, gx  = np.gradient(dem_arr, dy_m, dx_m)
    slope   = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    LAYERS["slope"] = slope
    if save and _vars.EXPORT_RASTERS:
        save_raster(slope, ext, os.path.join("rasters", "slope.tif"))
    return slope, ext

def derive_aspect(dem_arr=None, ext=None, save=True):
    """Calcule l'aspect (degrés, 0=N sens horaire) depuis le DEM local. Retourne (array, ext)."""
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT
    h, w    = dem_arr.shape
    dx_deg  = (ext[1] - ext[0]) / w
    dy_deg  = (ext[3] - ext[2]) / h
    lat_c   = (ext[2] + ext[3]) / 2.0
    dx_m    = dx_deg * 111320.0 * math.cos(math.radians(lat_c))
    dy_m    = dy_deg * 111320.0
    gy, gx  = np.gradient(dem_arr, dy_m, dx_m)
    aspect  = (np.degrees(np.arctan2(-gx, gy)) % 360).astype("float32")
    LAYERS["aspect"] = aspect
    if save and _vars.EXPORT_RASTERS:
        save_raster(aspect, ext, os.path.join("rasters", "aspect.tif"))
    return aspect, ext

def derive_curvature(dem_arr=None, ext=None, save=True):
    """Calcule la courbure (Laplacien 3x3) depuis le DEM local. Retourne (array, ext)."""
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT
    from scipy.ndimage import laplace
    curv = laplace(dem_arr).astype("float32")
    LAYERS["curvature"] = curv
    if save and _vars.EXPORT_RASTERS:
        save_raster(curv, ext, os.path.join("rasters", "curvature.tif"))
    return curv, ext

def derive_twi(dem_arr=None, ext=None, save=True):
    """
    Calcule le TWI (ln(a/tanB)) via pysheds D8 depuis le DEM local.
    Retourne (array, ext).
    """
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT
    g  = _get_pysheds_grid(dem_arr, ext)
    acc = np.asarray(g["acc"], dtype="float64")
    dem_r = np.asarray(g["dem"], dtype="float64")
    aff  = g["acc"].affine
    cell_m = abs(aff.a) * 111320.0
    gy_d, gx_d = np.gradient(dem_r, cell_m)
    tanb = np.maximum(np.hypot(gx_d, gy_d), 1e-3)
    a    = (acc + 1.0) * cell_m
    twi  = np.log(a / tanb).astype("float32")
    twi[np.isnan(dem_arr)] = np.nan
    LAYERS["twi"] = twi
    if save and _vars.EXPORT_RASTERS:
        save_raster(twi, ext, os.path.join("rasters", "twi.tif"))
    return twi, ext

def derive_flow_accumulation(dem_arr=None, ext=None, save=True):
    """Calcule l'accumulation de flux (km²) via pysheds D8. Retourne (array, ext)."""
    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT
    g   = _get_pysheds_grid(dem_arr, ext)
    acc = np.asarray(g["acc"], dtype="float64")
    aff = g["acc"].affine
    cell_m = abs(aff.a) * 111320.0
    acc_km2 = (acc * (cell_m / 1000.0) ** 2).astype("float32")
    acc_km2[np.isnan(dem_arr)] = np.nan
    LAYERS["flow_accumulation"] = acc_km2
    if save and _vars.EXPORT_RASTERS:
        save_raster(acc_km2, ext, os.path.join("rasters", "flow_accumulation.tif"))
    return acc_km2, ext
