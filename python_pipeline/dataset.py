import torch
from torch.utils.data import Dataset
import numpy as np

class LandslideSusceptibilityDataset(Dataset):
    """
    A custom PyTorch Dataset that manages multi-modal inputs for hybrid landslide susceptibility prediction.
    Each sample contains:
    1. Spatial/Topographic patch (e.g., 15x15 window around the cell) representing terrain parameters:
       - Layer 0: Elevation (DEM)
       - Layer 1: Slope angle
       - Layer 2: Aspect direction
       - Layer 3: Normalized Difference Vegetation Index (NDVI)
       - Layer 4: SAR Coherence loss (decorrelation)
       - Layer 5: Lithological geology class (encoded index)
    2. Dynamic Seismic features (vector):
       - Peak Ground Acceleration (PGA)
       - Arias Intensity (Ia)
       - Vs30 Soil stiffness parameter
    3. Target Label:
       - Binary value (1: Landslide triggered, 0: Stable terrain)
    """
    def __init__(self, spatial_patches, seismic_features, labels):
        self.spatial_patches = torch.tensor(spatial_patches, dtype=torch.float32)
        self.seismic_features = torch.tensor(seismic_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spatial_patches[idx], self.seismic_features[idx], self.labels[idx]


def generate_synthetic_data(num_samples=1000, patch_size=15):
    """
    Generates synthetic, physically-consistent data to train and validate the hybrid model.
    This simulates realistic geographical and seismic variables:
    
    Physics-based correlation logic:
    - Steep slopes (>30 degrees) + high PGA (>0.4g) + low vegetation (NDVI < 0.2) 
      + high SAR decorrelation -> high probability of landslide.
    - Flat terrain (<10 degrees) or zero ground motion -> extremely low probability.
    """
    np.random.seed(42)  # For reproducibility
    
    # 6 channels: Elevation, Slope, Aspect, NDVI, SAR Coherence loss, Lithology
    spatial_patches = np.zeros((num_samples, 6, patch_size, patch_size), dtype=np.float32)
    
    # 3 features: PGA, Arias Intensity, Vs30
    seismic_features = np.zeros((num_samples, 3), dtype=np.float32)
    
    labels = np.zeros(num_samples, dtype=np.float32)
    
    half_p = patch_size // 2
    
    for i in range(num_samples):
        # 1. Generate local terrain parameters
        # Elevation: base elevation from 500m to 2500m with spatial gradients
        elevation_base = np.random.uniform(500, 2500)
        x_grad, y_grad = np.random.uniform(-10, 10), np.random.uniform(-10, 10)
        
        # Creating coordinate grid for patch
        x, y = np.meshgrid(np.arange(patch_size) - half_p, np.arange(patch_size) - half_p)
        elev_patch = elevation_base + x_grad * x + y_grad * y
        
        # Slope: average slope 0 to 50 degrees
        slope_base = np.random.uniform(0, 50)
        slope_patch = np.clip(slope_base + np.random.normal(0, 2, size=(patch_size, patch_size)), 0, 90)
        
        # Aspect: 0 to 360 degrees
        aspect_base = np.random.uniform(0, 360)
        aspect_patch = (aspect_base + np.random.normal(0, 10, size=(patch_size, patch_size))) % 360
        
        # NDVI: vegetation density (0.0: barren/rock, 0.8: dense forest)
        ndvi_base = np.random.uniform(0.0, 0.8)
        ndvi_patch = np.clip(ndvi_base + np.random.normal(0, 0.05, size=(patch_size, patch_size)), -1.0, 1.0)
        
        # SAR Coherence loss: represents surface changes. If there is a landslide, it increases.
        # Initialize with baseline noise, we will adjust this if a landslide is triggered.
        sar_loss_patch = np.clip(np.random.normal(0.1, 0.05, size=(patch_size, patch_size)), 0.0, 1.0)
        
        # Lithology: 3 categories (0: Hard Granite, 1: Mixed Sandstone, 2: Loose Clay/Colluvium)
        lithology_base = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        lithology_patch = np.full((patch_size, patch_size), lithology_base, dtype=np.float32)
        
        # Combine channels
        spatial_patches[i, 0] = elev_patch / 3000.0  # Normalize elevation
        spatial_patches[i, 1] = slope_patch / 90.0    # Normalize slope
        spatial_patches[i, 2] = aspect_patch / 360.0  # Normalize aspect
        spatial_patches[i, 3] = ndvi_patch            # NDVI is already normalized
        spatial_patches[i, 5] = lithology_patch / 2.0  # Lithology index normalized
        
        # 2. Generate seismic inputs (for the central grid point)
        # PGA: 0 to 1.5g
        pga = np.random.uniform(0.0, 1.5)
        # Arias Intensity: correlates with PGA and magnitude
        arias = np.clip(pga**1.8 * np.random.uniform(5.0, 15.0), 0.0, 30.0)
        # Vs30: shear wave velocity in upper 30m (150 m/s soft to 1200 m/s hard rock)
        vs30 = np.random.uniform(150.0, 1200.0)
        
        seismic_features[i, 0] = pga
        seismic_features[i, 1] = arias
        seismic_features[i, 2] = vs30 / 1200.0  # Normalize Vs30
        
        # 3. Physics-based susceptibility logic to determine landslide label
        # Central cell properties
        center_slope = slope_patch[half_p, half_p]
        center_ndvi = ndvi_patch[half_p, half_p]
        
        # Calculate susceptibility score
        # Landslides need slope, lack of vegetation structure, and high dynamic seismic shaking
        slope_term = (center_slope / 45.0) ** 2  # Exponential risk with slope angle
        vegetation_term = 1.0 - center_ndvi      # Lower vegetation = higher risk
        seismic_term = pga * 2.5 + (arias / 15.0) # Shaking and energy input
        
        # Lithology multiplier: clay (index 2) is highly unstable, granite (index 0) is stable
        lith_multiplier = 0.5 if lithology_base == 0 else (1.0 if lithology_base == 1 else 1.5)
        
        # Combine into probability score
        score = (0.3 * slope_term + 0.1 * vegetation_term + 0.6 * seismic_term) * lith_multiplier
        # Map score to probability via sigmoid
        probability = 1.0 / (1.0 + np.exp(-3.0 * (score - 1.2)))
        
        # Decide label based on probability
        label = 1 if np.random.uniform(0, 1) < probability else 0
        labels[i] = label
        
        # Update SAR Coherence loss in the spatial patch based on landslide status
        # Landslides disrupt surface geometry causing SAR decorrelation (coherence loss)
        if label == 1:
            sar_loss_patch = np.clip(np.random.normal(0.7, 0.15, size=(patch_size, patch_size)), 0.2, 1.0)
        spatial_patches[i, 4] = sar_loss_patch
        
    return spatial_patches, seismic_features, labels
