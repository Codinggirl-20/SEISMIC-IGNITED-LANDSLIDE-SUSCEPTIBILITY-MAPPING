import numpy as np

def calculate_pga(magnitude, distance_km, depth_km, site_class='C'):
    """
    Estimates Peak Ground Acceleration (PGA) in units of g (acceleration due to gravity)
    using a standard Ground Motion Prediction Equation (GMPE) framework.
    
    Formula based on a simplified Joyner-Boore (1981) and Campbell (1997) hybrid relation:
    ln(PGA) = c1 + c2*M + c3*ln(sqrt(R^2 + d^2)) + c4*sqrt(R^2 + d^2) + site_amplification
    
    Parameters:
    -----------
    magnitude : float
        Moment magnitude of the earthquake (Mw), typical range 4.0 - 9.0.
    distance_km : float or np.ndarray
        Distance from the site to the earthquake source (epicentral or rupture distance).
    depth_km : float
        Focal depth of the earthquake in km.
    site_class : str
        NEHRP soil classification:
        'A' - Hard rock (Vs30 > 1500 m/s) -> low amplification
        'B' - Rock (760 < Vs30 <= 1500 m/s) -> baseline
        'C' - Very dense soil and soft rock (360 < Vs30 <= 760 m/s) -> moderate amplification
        'D' - Stiff soil (180 < Vs30 <= 360 m/s) -> high amplification
        'E' - Soft clay (Vs30 <= 180 m/s) -> very high amplification/liquefaction risk
    """
    # Empirically derived coefficients for PGA (in g)
    c1 = -3.512
    c2 = 0.904
    c3 = -1.328
    c4 = -0.0032
    
    # Effective distance combining path distance and depth
    r_eff = np.sqrt(distance_km**2 + depth_km**2)
    
    # Logarithmic PGA calculation
    ln_pga = c1 + c2 * magnitude + c3 * np.log(r_eff) + c4 * r_eff
    pga_rock = np.exp(ln_pga)
    
    # Soil amplification factor based on NEHRP site classes
    site_factors = {
        'A': 0.75,
        'B': 1.0,
        'C': 1.25,
        'D': 1.55,
        'E': 1.85
    }
    amplification = site_factors.get(site_class, 1.0)
    
    # Return calculated PGA (bounded by physically realistic limits)
    pga = pga_rock * amplification
    return np.clip(pga, 1e-4, 2.5)  # Cap at 2.5g for physical realism


def calculate_arias_intensity(magnitude, distance_km, depth_km, site_class='C'):
    """
    Estimates Arias Intensity (Ia) in m/s, which measures the total seismic energy
    absorbed by structures and slopes over the duration of the ground motion.
    
    Arias Intensity is highly correlated with landslide triggering threshold.
    Empirical relation based on Travasarou et al. (2003):
    ln(Ia) = c1 + c2*(M - 6) + c3*ln(sqrt(R^2 + d^2)) + site_term
    """
    c1 = 2.82
    c2 = 2.05
    c3 = -2.85
    
    r_eff = np.sqrt(distance_km**2 + depth_km**2)
    
    # Core empirical regression
    ln_ia = c1 + c2 * (magnitude - 6.0) + c3 * np.log(r_eff)
    ia_rock = np.exp(ln_ia)
    
    # Site amplification factors for energy intensity
    site_factors = {
        'A': 0.60,
        'B': 1.0,
        'C': 1.35,
        'D': 1.70,
        'E': 2.10
    }
    amplification = site_factors.get(site_class, 1.0)
    
    ia = ia_rock * amplification
    return np.clip(ia, 1e-5, 30.0)  # m/s


def compute_ndvi(nir_band, red_band):
    """
    Computes Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    Valid range is [-1, 1]. Positive values indicate vegetation.
    """
    # Prevent division by zero
    denominator = nir_band + red_band
    # Replace zeros with a tiny value to prevent NaN output
    if isinstance(denominator, np.ndarray):
        denominator = np.where(denominator == 0, 1e-6, denominator)
    else:
        if denominator == 0:
            denominator = 1e-6
            
    ndvi = (nir_band - red_band) / denominator
    return np.clip(ndvi, -1.0, 1.0)


def compute_sar_coherence_loss(coherence_pre, coherence_post):
    """
    Computes the loss of SAR coherence (decorrelation).
    Landslides destroy surface coherence (e.g. vegetation striping or soil movement),
    causing a significant drop in SAR coherence between pre- and post-event pairs.
    
    Formula: Delta_Coherence = Coherence_Pre - Coherence_Post
    """
    delta_coherence = coherence_pre - coherence_post
    # Bounded between 0 (no decorrelation) and 1 (complete decorrelation)
    return np.clip(delta_coherence, 0.0, 1.0)
