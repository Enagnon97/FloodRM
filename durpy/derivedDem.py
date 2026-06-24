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

def derive_drainage_density(dem_arr=None, ext=None, save=True,
                            threshold_km2=5.0, method="km/km2", window_km=2.0):
    """
    Densité de drainage depuis l'accumulation de flux locale.
    method : "km/km2"       → longueur cours d'eau / aire fenêtre circulaire
             "fraction_5km" → fraction de pixels-cours-d'eau dans fenêtre 5 km
    Utilise fftconvolve (FFT) pour les grands noyaux circulaires.
    """
    from scipy.signal import fftconvolve

    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT

    if "flow_accumulation" in LAYERS:
        facc = LAYERS["flow_accumulation"]
    else:
        facc, ext = derive_flow_accumulation(dem_arr, ext, save=False)

    h, w     = facc.shape
    dx_deg   = (ext[1] - ext[0]) / w
    dy_deg   = (ext[3] - ext[2]) / h
    lat_c    = (ext[2] + ext[3]) / 2.0
    cell_x_m = dx_deg * 111320.0 * math.cos(math.radians(lat_c))
    cell_y_m = dy_deg * 111320.0
    cell_m   = (cell_x_m + cell_y_m) / 2.0
    cell_km  = cell_m / 1000.0

    streams = np.where(np.isnan(facc), 0.0, (facc >= threshold_km2).astype("float64"))

    if method == "fraction_5km":
        r_pix  = max(1, int(5000.0 / cell_m))
        yi, xi = np.ogrid[-r_pix:r_pix+1, -r_pix:r_pix+1]
        kernel = (xi**2 + yi**2 <= r_pix**2).astype("float64")
        kernel /= kernel.sum()
        dd = fftconvolve(streams, kernel, mode="same").astype("float32")
    else:  # km/km2
        r_pix  = max(1, int(window_km * 1000.0 / cell_m))
        yi, xi = np.ogrid[-r_pix:r_pix+1, -r_pix:r_pix+1]
        kernel = (xi**2 + yi**2 <= r_pix**2).astype("float64")
        win_area_km2 = math.pi * (window_km ** 2)
        dd = (fftconvolve(streams * cell_km, kernel, mode="same")
              / win_area_km2).astype("float32")

    dd[np.isnan(dem_arr)] = np.nan
    LAYERS["drainage_density"] = dd
    if save and _vars.EXPORT_RASTERS:
        save_raster(dd, ext, os.path.join("rasters", "drainage_density.tif"))
    return dd, ext


def derive_dist_to_river(dem_arr=None, ext=None, save=True,
                         threshold_km2=5.0, max_dist_km=30.0):
    """
    Distance euclidienne aux cours d'eau (mètres), capée à max_dist_km.
    Équivalent local du fastDistanceTransform GEE.
    """
    from scipy.ndimage import distance_transform_edt

    if dem_arr is None:
        dem_arr, ext = fetch_array_local("elevation")
    if ext is None:
        ext = _vars._DEM_EXT

    if "flow_accumulation" in LAYERS:
        facc = LAYERS["flow_accumulation"]
    else:
        facc, ext = derive_flow_accumulation(dem_arr, ext, save=False)

    h, w     = facc.shape
    dx_deg   = (ext[1] - ext[0]) / w
    dy_deg   = (ext[3] - ext[2]) / h
    lat_c    = (ext[2] + ext[3]) / 2.0
    cell_x_m = dx_deg * 111320.0 * math.cos(math.radians(lat_c))
    cell_y_m = dy_deg * 111320.0

    # Pixels non-cours-d'eau = là où la distance est calculée
    streams    = (~np.isnan(facc)) & (facc >= threshold_km2)
    not_stream = ~streams

    # sampling=[dy, dx] → distance en mètres réels (pixels non carrés pris en compte)
    dist_m = distance_transform_edt(not_stream, sampling=[cell_y_m, cell_x_m])
    dist_m = np.minimum(dist_m, max_dist_km * 1000.0).astype("float32")
    dist_m[np.isnan(dem_arr)] = np.nan

    LAYERS["dist_to_river"] = dist_m
    if save and _vars.EXPORT_RASTERS:
        save_raster(dist_m, ext, os.path.join("rasters", "dist_to_river.tif"))
    return dist_m, ext

def load_lulc(path, save=False):
    """
    Charge une couche LULC locale (GeoTIFF) et la stocke dans LAYERS["worldcover"].
    Les codes de classe sont définis dans durpy.variables.WC_LEGEND.
    Retourne (array int16, ext).
    """
    with rasterio.open(path) as src:
        a  = src.read(1)
        nd = src.nodata
        b  = src.bounds
        ext = [b.left, b.right, b.bottom, b.top]

    a = a.astype("float32")
    if nd is not None:
        a[a == float(nd)] = np.nan

    if _vars._DEM_ARRAY is None:
        raise RuntimeError(
            "DEM non initialisé : appelez init_local_engine(DEM_PATH) avant load_lulc()."
        )

    # Ré-échantillonne au format DEM (nearest-neighbor : données catégorielles)
    from scipy.ndimage import zoom
    dem_h, dem_w = _vars._DEM_ARRAY.shape
    if a.shape != (dem_h, dem_w):
        zoom_y = dem_h / a.shape[0]
        zoom_x = dem_w / a.shape[1]
        a = zoom(a, (zoom_y, zoom_x), order=0)  # order=0 = nearest-neighbor

    # Applique le masque DEM
    a[np.isnan(_vars._DEM_ARRAY)] = np.nan

    LAYERS["worldcover"] = a
    return a, ext


def load_rain(path, save=False):
    """
    Charge un fichier NetCDF de pluies quotidiennes (CHIRPS ou similaire)
    et calcule la pluie annuelle moyenne (mm/an).
    Rééchantillonne à la résolution du DEM par interpolation bilinéaire.
    Stocke dans LAYERS["rain_mm_yr"]. Retourne (array float32, ext).
    """
    import netCDF4 as nc
    from scipy.ndimage import zoom as nd_zoom

    with nc.Dataset(path) as ds:
        # Détecte la variable de précipitation (premier champ 3D)
        precip_var = None
        for v in ds.variables:
            if ds.variables[v].ndim == 3:
                precip_var = v
                break
        if precip_var is None:
            raise RuntimeError("Aucune variable 3D (time, lat, lon) trouvée dans le fichier NetCDF.")

        data  = ds.variables[precip_var][:]          # (time, lat, lon)
        lats  = ds.variables['lat'][:]
        lons  = ds.variables['lon'][:]
        ndays = data.shape[0]

    # Pluie annuelle moyenne : somme totale / nombre d'années
    n_years  = ndays / 365
    rain_sum = np.nansum(data, axis=0)               # (lat, lon)
    rain_yr  = (rain_sum / n_years).astype("float32")

    # Lat CHIRPS est souvent du nord au sud → vérifier et réordonner si besoin
    if lats[0] > lats[-1]:
        rain_yr = rain_yr[::-1, :]
        lats    = lats[::-1]

    ext_rain = [float(lons.min()), float(lons.max()),
                float(lats.min()), float(lats.max())]

    # Rééchantillonne à la résolution du DEM si nécessaire
    if _vars._DEM_ARRAY is None:
        raise RuntimeError(
            "DEM non initialisé : appelez init_local_engine(DEM_PATH) avant load_rain()."
        )
    dem_h, dem_w = _vars._DEM_ARRAY.shape
    zoom_y = dem_h / rain_yr.shape[0]
    zoom_x = dem_w / rain_yr.shape[1]
    rain_yr = nd_zoom(rain_yr, (zoom_y, zoom_x), order=1)  # bilinéaire
    ext = _vars._DEM_EXT
    # Applique le masque DEM
    rain_yr[np.isnan(_vars._DEM_ARRAY)] = np.nan

    rain_yr = rain_yr.astype("float32")
    LAYERS["rain_mm_yr"] = rain_yr

    mn = float(np.nanmin(rain_yr))
    mx = float(np.nanmax(rain_yr))
    print(f"Pluie annuelle moyenne chargee : min={mn:.1f} mm/an | max={mx:.1f} mm/an")

    if save and _vars.EXPORT_RASTERS:
        save_raster(rain_yr, ext, os.path.join("rasters", "rain_mm_yr.tif"))
    return rain_yr, ext

def load_rain_v2(path, save=False):
    """
    Charge CHIRPS quotidien via xarray.
    Cumul annuel (mm/an) = somme journalière par année, puis moyenne inter-annuelle.
    min_count=1 garantit NaN là où il n'y a aucune donnée valide (pas de 0 parasite).
    """
    import xarray as xr
    from scipy.ndimage import zoom as nd_zoom
    #from durpy import _vars
    from durpy.variables import LAYERS

    if _vars._DEM_ARRAY is None:
        raise RuntimeError("Appelez init_local_engine(DEM_PATH) avant load_rain_v2().")

    ds = xr.open_dataset(path, chunks={})  # lazy load

    # Détecte la variable de précipitation (premier champ 3D avec dim 'time')
    precip_var = next(
        (v for v in ds.data_vars if ds[v].ndim == 3 and "time" in ds[v].dims),
        None
    )
    if precip_var is None:
        raise RuntimeError("Aucune variable 3D (time, lat, lon) trouvée dans le fichier NetCDF.")

    da = ds[precip_var]  # (time, lat, lon)

    # 1. Cumul annuel (mm/an) — min_count=1 → NaN si aucun jour valide dans l'année
    annual = da.resample(time="YE").sum(skipna=True, min_count=1)

    # 2. Moyenne inter-annuelle des cumuls annuels
    rain_yr = annual.mean(dim="time", skipna=True).values.astype("float32")  # (lat, lon)

    ds.close()

    # Réorientation lat si nécessaire (CHIRPS : N→S parfois)
    lats = ds["lat"].values
    if lats[0] > lats[-1]:
        rain_yr = rain_yr[::-1, :]

    # Ré-échantillonnage au format DEM (interpolation bilinéaire)
    dem_h, dem_w = _vars._DEM_ARRAY.shape
    if rain_yr.shape != (dem_h, dem_w):
        zoom_y = dem_h / rain_yr.shape[0]
        zoom_x = dem_w / rain_yr.shape[1]
        rain_yr = nd_zoom(rain_yr, (zoom_y, zoom_x), order=1)

    # Masque DEM
    rain_yr[np.isnan(_vars._DEM_ARRAY)] = np.nan

    rain_yr = rain_yr.astype("float32")
    LAYERS["rain_mm_yr"] = rain_yr

    mn = float(np.nanmin(rain_yr))
    mx = float(np.nanmax(rain_yr))
    print(f"Pluie annuelle cumul moyen : min={mn:.1f} mm/an | max={mx:.1f} mm/an")
    return rain_yr, _vars._DEM_EXT

