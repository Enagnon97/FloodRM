import math
import numpy as np
import pandas as pd

from durpy.variables import LAYERS

SAATY_RI = {
    1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51,
    12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}

# =============================================================================
# NORMALISATION
# =============================================================================

def normalize_layer(arr, method="minmax"):
    """
    Normalise un tableau numpy vers [0,1]. Les NaN sont préservés.
    Méthodes : minmax | robust | zscore | fuzzy_linear | fuzzy_sigmoidal
    """
    a = arr.astype("float64")
    mask = np.isnan(a)

    if method == "minmax":
        mn, mx = np.nanmin(a), np.nanmax(a)
        if mx <= mn:
            return np.zeros_like(arr, dtype="float32")
        out = (a - mn) / (mx - mn)

    elif method == "robust":
        p2, p98 = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
        if p98 <= p2:
            return normalize_layer(arr, "minmax")
        out = np.clip((a - p2) / (p98 - p2), 0, 1)

    elif method == "zscore":
        mu, sd = np.nanmean(a), np.nanstd(a)
        if sd == 0:
            return np.zeros_like(arr, dtype="float32")
        out = np.clip((a - mu) / sd, -3, 3)
        out = (out + 3) / 6

    elif method == "fuzzy_linear":
        p10, p90 = np.nanpercentile(a, 10), np.nanpercentile(a, 90)
        if p90 <= p10:
            return normalize_layer(arr, "minmax")
        out = np.clip((a - p10) / (p90 - p10), 0, 1)

    elif method == "fuzzy_sigmoidal":
        p5, p95 = np.nanpercentile(a, 5), np.nanpercentile(a, 95)
        if p95 <= p5:
            return normalize_layer(arr, "minmax")
        xr = np.clip((a - p5) / (p95 - p5), 0, 1)
        out = 0.5 * (1 - np.cos(math.pi * xr))

    else:
        raise ValueError(f"Méthode inconnue : {method}. "
                         "Choix : minmax, robust, zscore, fuzzy_linear, fuzzy_sigmoidal")

    result = np.where(mask, np.nan, np.clip(out, 0, 1)).astype("float32")
    return result


def normalize_all(keys, method="minmax"):
    """
    Normalise toutes les couches listées dans keys.
    Stocke LAYERS["norm_<k>"] pour chaque clé.
    Retourne un DataFrame des statistiques.
    """
    rows = []
    for k in keys:
        if k not in LAYERS:
            print(f"  skip {k} (absent de LAYERS)")
            continue
        arr = LAYERS[k]
        LAYERS["norm_" + k] = normalize_layer(arr, method)
        rows.append({
            "criterion": k,
            "min":    round(float(np.nanmin(arr)), 4),
            "max":    round(float(np.nanmax(arr)), 4),
            "mean":   round(float(np.nanmean(arr)), 4),
            "std":    round(float(np.nanstd(arr)), 4),
            "method": method,
        })
        print(f"  ✓ norm_{k}")
    return pd.DataFrame(rows)


# =============================================================================
# MATRICE AHP & POIDS
# =============================================================================

def build_ahp_matrix(AHP_PAIRS, AHP_CRITERIA, active_keys):
    """
    Construit la sous-matrice de Saaty pour les critères actifs.
    Retourne (A, labels).
    """
    avail = [i for i, k in enumerate(AHP_CRITERIA) if k in active_keys]
    labels = [AHP_CRITERIA[i] for i in avail]
    n = len(labels)
    A = np.ones((n, n))
    for (gi, gj), v in AHP_PAIRS.items():
        ci, cj = AHP_CRITERIA[gi], AHP_CRITERIA[gj]
        if ci in labels and cj in labels:
            li, lj = labels.index(ci), labels.index(cj)
            A[li, lj] = v
            A[lj, li] = 1.0 / v
    return A, labels


def ahp_solve(A):
    """
    Résout la matrice AHP par vecteur propre principal.
    Retourne (weights, weights_rgmm, lambda_max, CI, RI, CR).
    weights_rgmm = poids par moyenne géométrique des lignes (RGMM).
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    val, vec = np.linalg.eig(A)
    k = int(np.argmax(val.real))
    lmax = float(val[k].real)
    w = np.abs(vec[:, k].real)
    w = w / w.sum()
    g = np.exp(np.mean(np.log(A), axis=1))
    w_rgmm = g / g.sum()
    CI = (lmax - n) / (n - 1) if n > 2 else 0.0
    RI = SAATY_RI.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return w, w_rgmm, lmax, CI, RI, CR


def compute_weights(AHP_PAIRS, AHP_CRITERIA, AHP_INCLUDE, SCORES):
    """
    Calcule les poids AHP pour les critères actifs présents dans SCORES.
    Affiche lambda_max, CI, RI, CR.
    Retourne dict {criterion: weight}.
    """
    active_keys = [k for k in AHP_CRITERIA
                   if AHP_INCLUDE.get(k, True) and k in SCORES]
    A, labels = build_ahp_matrix(AHP_PAIRS, AHP_CRITERIA, active_keys)
    w, w_rgmm, lmax, CI, RI, CR = ahp_solve(A)

    verdict = "✅ consistant (CR < 0.10)" if CR < 0.10 else "⚠️  INCONSISTANT — réviser les jugements"
    print(f"Critères actifs ({len(labels)}) : {', '.join(labels)}")
    print(f"λ_max = {lmax:.3f} | CI = {CI:.4f} | RI = {RI:.2f} | CR = {CR:.4f} → {verdict}")

    df = pd.DataFrame({
        "criterion":         labels,
        "weight_eigenvector": np.round(w, 4),
        "weight_RGMM":       np.round(w_rgmm, 4),
        "weight_%":          np.round(w * 100, 2),
    }).sort_values("weight_eigenvector", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    print(df.to_string(index=False))

    return {labels[i]: float(w[i]) for i in range(len(labels))}


# =============================================================================
# RECLASSIFICATION
# =============================================================================

def quantile_breaks(arr, qs=(20, 40, 60, 80)):
    """Calcule 4 seuils de reclassification par percentiles."""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return [0.2, 0.4, 0.6, 0.8]
    breaks = [float(np.percentile(vals, q)) for q in qs]
    eps = float(np.ptp(vals)) * 1e-6 if np.ptp(vals) > 0 else 1e-6
    for i in range(1, len(breaks)):
        if breaks[i] <= breaks[i - 1]:
            breaks[i] = breaks[i - 1] + eps
    return breaks


def reclass_continuous(arr, breaks, scores):
    """Reclassifie un tableau continu avec des seuils fixes → scores 1–5."""
    out = np.full(arr.shape, float(scores[0]), dtype="float64")
    for b, s in zip(breaks, scores[1:]):
        out = np.where(arr > b, float(s), out)
    return np.where(np.isnan(arr), np.nan, out).astype("float32")


def reclass_categorical(arr, mapping, default=0):
    """Reclassifie un tableau catégoriel (ex : codes LULC → scores 1–5)."""
    out = np.full(arr.shape, float(default), dtype="float64")
    for code, score in mapping.items():
        out = np.where(np.isclose(arr, code, equal_nan=False), float(score), out)
    out = np.where(np.isnan(arr), np.nan, out)
    out = np.where((out == 0) & (~np.isnan(out)), np.nan, out)
    return out.astype("float32")


def reclass_aspect(arr):
    """
    Aspect (degrés) → score 1–5 via la northness (cosinus).
    Nord (0°/360°) = versant plus humide → score 5 (risque élevé).
    """
    rad = np.radians(arr.astype("float64"))
    northness = np.cos(rad)                  # [-1, 1]
    out = (northness + 1) / 2.0 * 4.0 + 1.0 # [1, 5]
    return np.where(np.isnan(arr), np.nan, out).astype("float32")


def make_score(k, reclass_cfg, use_norm=False):
    """
    Route vers la bonne fonction de reclassification pour le facteur k.
    Retourne un array de scores 1–5, ou None si k absent de LAYERS.
    """
    if k not in LAYERS:
        return None
    sp = reclass_cfg[k]
    kind = sp["kind"]

    if use_norm and ("norm_" + k) in LAYERS and kind in ("continuous", "quantile"):
        arr = LAYERS["norm_" + k]
        kind = "quantile"
        sp = dict(sp)
    else:
        arr = LAYERS[k]

    if kind == "aspect":
        return reclass_aspect(arr)
    if kind == "categorical":
        return reclass_categorical(arr, sp["mapping"])
    if kind == "continuous":
        return reclass_continuous(arr, sp["breaks"], sp["scores"])
    if kind == "quantile":
        br = quantile_breaks(arr)
        return reclass_continuous(arr, br, sp.get("scores", [1, 2, 3, 4, 5]))
    return None


def compute_scores(reclass_cfg, ahp_include, use_norm=False):
    """
    Calcule les cartes de scores (1–5) pour tous les facteurs actifs présents dans LAYERS.
    Retourne dict {factor_key: score_array}.
    """
    SCORES = {}
    for k in reclass_cfg:
        if not ahp_include.get(k, True):
            continue
        if k not in LAYERS:
            print(f"  skip {k} (absent de LAYERS)")
            continue
        score = make_score(k, reclass_cfg, use_norm)
        if score is not None:
            SCORES[k] = score
            print(f"  ✓ score_{k}")
    return SCORES


# =============================================================================
# INDICE DE RISQUE D'INONDATION
# =============================================================================

def compute_flood_index(SCORES, weights):
    """
    Somme pondérée des scores → Flood Index (FIPS).
    Les poids sont renormalisés sur les facteurs actifs.
    Retourne array FIPS (valeurs ~1–5).
    """
    active = [k for k in weights if k in SCORES]
    if not active:
        raise RuntimeError("Aucun facteur actif dans SCORES.")
    wsum = sum(weights[k] for k in active) or 1.0
    shape = next(SCORES[k] for k in active).shape
    bad = {k: SCORES[k].shape for k in active if SCORES[k].shape != shape}
    if bad:
        raise ValueError(
            f"Formes incompatibles dans SCORES (référence {shape}) : {bad}.\n"
            "Assurez-vous d'appeler init_local_engine() avant load_rain()/load_lulc(), "
            "puis relancez toutes les cellules dans l'ordre."
        )
    fips = np.zeros(shape, dtype="float64")
    for k in active:
        s = np.where(np.isnan(SCORES[k]), 0.0, SCORES[k])
        fips += s * (weights[k] / wsum)
    ref = SCORES[active[0]]
    fips = np.where(np.isnan(ref), np.nan, fips)
    return fips.astype("float32")


def classify_flood_index(fips, method="equal", breaks=None):
    """
    Classe le FIPS en 5 zones de risque d'inondation.
    method "equal"    → seuils [1.8, 2.6, 3.4, 4.2]
    method "quantile" → seuils calculés sur la distribution du FIPS
    Retourne (zones_array, breaks_used).
    """
    if breaks is None:
        breaks = [1.8, 2.6, 3.4, 4.2] if method == "equal" else quantile_breaks(fips)
    zones = reclass_continuous(fips, breaks, [1, 2, 3, 4, 5])
    print("Seuils zones :", [round(b, 3) for b in breaks])
    return zones.astype("float32"), breaks


# =============================================================================
# ANALYSE DE SENSIBILITÉ
# =============================================================================

def sensitivity_analysis(SCORES, weights, fips):
    """
    Supprime chaque facteur un par un, recalcule le FIPS sans lui,
    mesure l'écart absolu moyen (MAD).
    Retourne DataFrame trié par sensibilité décroissante.
    """
    active = [k for k in weights if k in SCORES]
    wsum = sum(weights[k] for k in active) or 1.0
    ref_mask = np.isnan(SCORES[active[0]])
    rows = []
    for k in active:
        wk = weights[k] / wsum
        w_rest = wsum - weights[k]
        if w_rest <= 0:
            mad = float("nan")
        else:
            fips_k = np.zeros_like(fips, dtype="float64")
            for j in active:
                if j == k:
                    continue
                s = np.where(np.isnan(SCORES[j]), 0.0, SCORES[j])
                fips_k += s * (weights[j] / w_rest)
            fips_k = np.where(ref_mask, np.nan, fips_k)
            mad = float(np.nanmean(np.abs(fips.astype("float64") - fips_k)))
        rows.append({"criterion": k, "weight_%": round(wk * 100, 2), "MAD_FIPS": round(mad, 4)})
    df = pd.DataFrame(rows).sort_values("MAD_FIPS", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
