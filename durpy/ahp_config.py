# =============================================================================
# AHP CONFIG — Flood Susceptibility Mapping  (13 facteurs)
# =============================================================================

# --- Score LULC (WorldCover) ---
WORLDCOVER_SCORE_MAP = {
    10:  2,   # Forêt/Arbres
    20:  2,   # Arbustes
    30:  3,   # Prairie
    40:  3,   # Cultures
    50:  4,   # Zones bâties
    60:  3,   # Sol nu / éparse
    70:  1,   # Neige / glace
    80:  5,   # Eau permanente
    90:  5,   # Zones humides
    95:  4,   # Mangroves
    100: 2,   # Mousses / lichens
}

# --- Règles de reclassification (scores 1-5) ---
# scores [5,4,3,2,1] = valeur basse → risque élevé
# scores [1,2,3,4,5] = valeur haute → risque élevé
RECLASS = {
    "elevation":        dict(kind="quantile",    scores=[5, 4, 3, 2, 1]),
    "slope":            dict(kind="continuous",  breaks=[2, 5, 10, 20],         scores=[5, 4, 3, 2, 1]),
    "aspect":           dict(kind="aspect"),
    "curvature":        dict(kind="quantile",    scores=[5, 4, 3, 2, 1]),
    "dist_to_river":    dict(kind="continuous",  breaks=[250, 500, 1000, 2000],  scores=[5, 4, 3, 2, 1]),
    "drainage_density": dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    "twi":              dict(kind="continuous",  breaks=[6, 9, 12, 16],          scores=[1, 2, 3, 4, 5]),
    "flow_accumulation":dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    "worldcover":       dict(kind="categorical", mapping=WORLDCOVER_SCORE_MAP),
    "rain_mm_yr":       dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    # Nouveaux facteurs
    "hand":             dict(kind="continuous",  breaks=[10, 20, 50, 100],       scores=[5, 4, 3, 2, 1]),
    "runoff":           dict(kind="quantile",    scores=[1, 2, 3, 4, 5]),
    "dist_to_road":     dict(kind="continuous",  breaks=[500, 1000, 2000, 4000], scores=[5, 4, 3, 2, 1]),
}

# --- Ordre des critères (indices de la matrice AHP) ---
AHP_CRITERIA = [
    "elevation",         # 0
    "slope",             # 1
    "aspect",            # 2
    "curvature",         # 3
    "dist_to_river",     # 4
    "drainage_density",  # 5
    "twi",               # 6
    "flow_accumulation", # 7
    "worldcover",        # 8
    "rain_mm_yr",        # 9
    "hand",              # 10  ← HAND (Height Above Nearest Drainage)
    "runoff",            # 11  ← Ruissellement SCS-CN
    "dist_to_road",      # 12  ← Distance aux routes OSM
]

# --- Activation des facteurs ---
# rain_mm_yr désactivé : le ruissellement SCS-CN (runoff) l'incorpore déjà
AHP_INCLUDE = {k: True for k in AHP_CRITERIA}
AHP_INCLUDE["rain_mm_yr"] = False

# --- Matrice de Saaty (13 critères) ---
# AHP_PAIRS[(i,j)] avec i>j : "critère i est x fois plus important que j"
# Lecture : (twi=6, elevation=0): 2 → twi 2× plus important qu'elevation
AHP_PAIRS = {
    # ── Paires originales (facteurs 0–9) ──────────────────────────────────────
    (1,  0): 1,          # slope ~ elevation
    (2,  0): 1/5,        # aspect << elevation
    (2,  1): 1/7,        # aspect << slope
    (3,  0): 1/3,        # curvature < elevation
    (3,  1): 1/3,        # curvature < slope
    (3,  2): 2,          # curvature > aspect
    (4,  0): 2,          # dist_to_river > elevation
    (4,  1): 1,          # dist_to_river ~ slope
    (4,  2): 7,          # dist_to_river >> aspect
    (4,  3): 5,          # dist_to_river >> curvature
    (5,  0): 1/2,        # drainage_density < elevation
    (5,  1): 1/2,        # drainage_density < slope
    (5,  2): 5,          # drainage_density >> aspect
    (5,  3): 3,          # drainage_density > curvature
    (5,  4): 1/3,        # drainage_density < dist_to_river
    (6,  0): 2,          # twi > elevation
    (6,  1): 2,          # twi > slope
    (6,  2): 9,          # twi >>> aspect
    (6,  3): 5,          # twi >> curvature
    (6,  4): 1,          # twi ~ dist_to_river
    (6,  5): 3,          # twi > drainage_density
    (7,  0): 2,          # flow_acc > elevation
    (7,  1): 1,          # flow_acc ~ slope
    (7,  2): 9,          # flow_acc >>> aspect
    (7,  3): 5,          # flow_acc >> curvature
    (7,  4): 1,          # flow_acc ~ dist_to_river
    (7,  5): 3,          # flow_acc > drainage_density
    (7,  6): 1,          # flow_acc ~ twi
    (8,  0): 1/3,        # worldcover < elevation
    (8,  1): 1/3,        # worldcover < slope
    (8,  2): 3,          # worldcover > aspect
    (8,  3): 2,          # worldcover > curvature
    (8,  4): 1/5,        # worldcover < dist_to_river
    (8,  5): 1/2,        # worldcover < drainage_density
    (8,  6): 1/5,        # worldcover < twi
    (8,  7): 1/5,        # worldcover < flow_acc
    (9,  0): 2,          # rain > elevation
    (9,  1): 2,          # rain > slope
    (9,  2): 7,          # rain >> aspect
    (9,  3): 5,          # rain >> curvature
    (9,  4): 1/2,        # rain < dist_to_river
    (9,  5): 2,          # rain > drainage_density
    (9,  6): 1/2,        # rain < twi
    (9,  7): 1/2,        # rain < flow_acc
    (9,  8): 3,          # rain > worldcover

    # ── HAND (10) — indicateur de hauteur au-dessus du drainage ──────────────
    (10, 0): 3,          # hand > elevation
    (10, 1): 2,          # hand > slope
    (10, 2): 7,          # hand >> aspect
    (10, 3): 5,          # hand >> curvature
    (10, 4): 1,          # hand ~ dist_to_river (deux mesures de proximité à l'eau)
    (10, 5): 3,          # hand > drainage_density
    (10, 6): 1,          # hand ~ twi (deux indices topographiques de l'eau)
    (10, 7): 1,          # hand ~ flow_acc
    (10, 8): 5,          # hand >> worldcover
    (10, 9): 2,          # hand > rain

    # ── Runoff SCS-CN (11) — ruissellement effectif ───────────────────────────
    (11, 0): 2,          # runoff > elevation
    (11, 1): 1,          # runoff ~ slope
    (11, 2): 7,          # runoff >> aspect
    (11, 3): 5,          # runoff >> curvature
    (11, 4): 1/2,        # runoff < dist_to_river
    (11, 5): 2,          # runoff > drainage_density
    (11, 6): 1/2,        # runoff < twi
    (11, 7): 1/2,        # runoff < flow_acc
    (11, 8): 3,          # runoff > worldcover
    (11, 9): 2,          # runoff > rain (mesure plus directe que la pluie brute)
    (11, 10): 1/2,       # runoff < hand

    # ── Distance aux routes OSM (12) ─────────────────────────────────────────
    (12, 0): 1/2,        # dist_road < elevation
    (12, 1): 1/2,        # dist_road < slope
    (12, 2): 3,          # dist_road > aspect
    (12, 3): 2,          # dist_road > curvature
    (12, 4): 1/5,        # dist_road << dist_to_river
    (12, 5): 1,          # dist_road ~ drainage_density
    (12, 6): 1/5,        # dist_road << twi
    (12, 7): 1/5,        # dist_road << flow_acc
    (12, 8): 1,          # dist_road ~ worldcover
    (12, 9): 1/3,        # dist_road < rain
    (12, 10): 1/5,       # dist_road << hand
    (12, 11): 1/3,       # dist_road < runoff
}

# --- Couches à normaliser (aspect, worldcover et rain exclus) ---
NORM_KEYS = [
    "elevation", "slope", "curvature", "dist_to_river",
    "drainage_density", "twi", "flow_accumulation",
    "hand", "runoff", "dist_to_road",
]
