# =============================================================================
# AHP CONFIG — Flood Risk Mapping
# Modifie ce fichier pour ajuster les règles de reclassification,
# la matrice de Saaty et les facteurs actifs.
# =============================================================================

# --- Score LULC (WorldCover) pour le risque d'inondation ---
# 5 = risque très élevé, 1 = risque très faible
WORLDCOVER_SCORE_MAP = {
    10:  2,   # Forêt/Arbres       → réduit le ruissellement
    20:  2,   # Arbustes
    30:  3,   # Prairie
    40:  3,   # Cultures
    50:  4,   # Zones bâties       → imperméable, ruissellement élevé
    60:  3,   # Sol nu / éparse
    70:  1,   # Neige / glace
    80:  5,   # Eau permanente     → risque maximal
    90:  5,   # Zones humides
    95:  4,   # Mangroves
    100: 2,   # Mousses / lichens
}

# --- Règles de reclassification par facteur ---
# kind : "continuous" (seuils fixes), "quantile" (seuils calculés),
#        "categorical" (codes → scores), "aspect" (formule cosinus)
# scores : direction du risque → [5,4,3,2,1] = valeur basse = risque élevé
#                                [1,2,3,4,5] = valeur haute = risque élevé
RECLASS = {
    "elevation":         dict(kind="quantile",    scores=[5, 4, 3, 2, 1]),
    "slope":             dict(kind="continuous",  breaks=[2, 5, 10, 20],        scores=[5, 4, 3, 2, 1]),
    "aspect":            dict(kind="aspect"),
    "curvature":         dict(kind="quantile",    scores=[5, 4, 3, 2, 1]),
    "dist_to_river":     dict(kind="continuous",  breaks=[250, 500, 1000, 2000], scores=[5, 4, 3, 2, 1]),
    "drainage_density":  dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    "twi":               dict(kind="continuous",  breaks=[6, 9, 12, 16],         scores=[1, 2, 3, 4, 5]),
    "flow_accumulation": dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    "worldcover":        dict(kind="categorical", mapping=WORLDCOVER_SCORE_MAP),
}

# --- Ordre des critères (indices de la matrice AHP) ---
AHP_CRITERIA = [
    "elevation",        # 0
    "slope",            # 1
    "aspect",           # 2
    "curvature",        # 3
    "dist_to_river",    # 4
    "drainage_density", # 5
    "twi",              # 6
    "flow_accumulation",# 7
    "worldcover",       # 8
]

# --- Activation des facteurs (mettre False pour exclure) ---
AHP_INCLUDE = {k: True for k in AHP_CRITERIA}

# --- Matrice de Saaty (flood risk) ---
# AHP_PAIRS[(i,j)] avec i>j : "le critère i est x fois plus important que j"
# Indices selon AHP_CRITERIA ci-dessus.
AHP_PAIRS = {
    (1, 0): 1,          # slope ~ elevation
    (2, 0): 1/5,        # aspect << elevation
    (2, 1): 1/7,        # aspect << slope
    (3, 0): 1/3,        # curvature < elevation
    (3, 1): 1/3,        # curvature < slope
    (3, 2): 2,          # curvature > aspect
    (4, 0): 2,          # dist_to_river > elevation
    (4, 1): 1,          # dist_to_river ~ slope
    (4, 2): 7,          # dist_to_river >> aspect
    (4, 3): 5,          # dist_to_river >> curvature
    (5, 0): 1/2,        # drainage_density < elevation
    (5, 1): 1/2,        # drainage_density < slope
    (5, 2): 5,          # drainage_density >> aspect
    (5, 3): 3,          # drainage_density > curvature
    (5, 4): 1/3,        # drainage_density < dist_to_river
    (6, 0): 2,          # twi > elevation
    (6, 1): 2,          # twi > slope
    (6, 2): 9,          # twi >>> aspect
    (6, 3): 5,          # twi >> curvature
    (6, 4): 1,          # twi ~ dist_to_river
    (6, 5): 3,          # twi > drainage_density
    (7, 0): 2,          # flow_accumulation > elevation
    (7, 1): 1,          # flow_accumulation ~ slope
    (7, 2): 9,          # flow_accumulation >>> aspect
    (7, 3): 5,          # flow_accumulation >> curvature
    (7, 4): 1,          # flow_accumulation ~ dist_to_river
    (7, 5): 3,          # flow_accumulation > drainage_density
    (7, 6): 1,          # flow_accumulation ~ twi
    (8, 0): 1/3,        # worldcover < elevation
    (8, 1): 1/3,        # worldcover < slope
    (8, 2): 3,          # worldcover > aspect
    (8, 3): 2,          # worldcover > curvature
    (8, 4): 1/5,        # worldcover < dist_to_river
    (8, 5): 1/2,        # worldcover < drainage_density
    (8, 6): 1/5,        # worldcover < twi
    (8, 7): 1/5,        # worldcover < flow_accumulation
}

# --- Couches à normaliser (aspect et worldcover exclus) ---
NORM_KEYS = [
    "elevation", "slope", "curvature", "dist_to_river",
    "drainage_density", "twi", "flow_accumulation",
]
