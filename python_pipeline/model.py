import torch
import torch.nn as nn

class HybridLandslideModel(nn.Module):
    """
    A multi-modal Hybrid Deep Learning architecture for landslide susceptibility mapping.
    
    It integrates two distinct streams:
    1. Spatial Stream (2D CNN): Processes local topographic and remote sensing patch matrices.
    2. Seismic Stream (MLP): Processes point-specific dynamic seismic ground motion parameters.
    
    The representations from both streams are fused to perform final susceptibility classification.
    """
    def __init__(self, spatial_channels=6, seismic_dim=3, patch_size=15):
        super(HybridLandslideModel, self).__init__()
        
        # --- Stream 1: Spatial/Topographic Feature Extractor (2D CNN) ---
        # Input shape: (Batch, 6, PatchSize, PatchSize)
        self.spatial_cnn = nn.Sequential(
            # Block 1: Capture fine textures
            nn.Conv2d(in_channels=spatial_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (Batch, 16, PatchSize/2, PatchSize/2)
            
            # Block 2: Extract intermediate spatial features
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Global Average Pooling flattens the spatial maps regardless of patch size
            nn.AdaptiveAvgPool2d((1, 1)) # Output: (Batch, 32, 1, 1)
        )
        
        # --- Stream 2: Dynamic Seismic Feature Extractor (MLP) ---
        # Input shape: (Batch, SeismicDim)
        self.seismic_mlp = nn.Sequential(
            nn.Linear(seismic_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        # --- Fusion & Classification Head ---
        # Concatenated feature size = CNN output channels (32) + MLP output features (16)
        fusion_dim = 32 + 16
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1) # Outputs raw logits. Apply sigmoid for probability calculation.
        )
        
    def forward(self, spatial_patch, seismic_vector):
        """
        Forward pass of the hybrid model.
        
        Parameters:
        -----------
        spatial_patch : torch.Tensor
            Geospatial raster channels, shape (Batch, Channels, Height, Width)
        seismic_vector : torch.Tensor
            Point ground motion vectors, shape (Batch, Features)
            
        Returns:
        --------
        logits : torch.Tensor
            Landslide susceptibility logit, shape (Batch, 1)
        """
        # 1. Process spatial data stream
        spatial_features = self.spatial_cnn(spatial_patch)
        spatial_features = spatial_features.view(spatial_features.size(0), -1) # Flatten to (Batch, 32)
        
        # 2. Process seismic data stream
        seismic_features = self.seismic_mlp(seismic_vector) # Shape: (Batch, 16)
        
        # 3. Multi-modal feature fusion
        fused_features = torch.cat((spatial_features, seismic_features), dim=1) # Shape: (Batch, 48)
        
        # 4. Binary classification logit
        logits = self.classifier(fused_features)
        return logits


if __name__ == "__main__":
    # Test execution block to verify shape alignment and initialization correctness
    print("Testing HybridLandslideModel initialization...")
    model = HybridLandslideModel()
    
    # Generate mock inputs corresponding to 4 sample grid points
    # 4 samples, 6 spatial channels, 15x15 pixel elevation/optical patch
    mock_spatial = torch.randn(4, 6, 15, 15)
    # 4 samples, 3 seismic parameters (PGA, Arias, Vs30)
    mock_seismic = torch.randn(4, 3)
    
    # Run mock inference
    logits = model(mock_spatial, mock_seismic)
    print(f"Input spatial tensor shape: {mock_spatial.shape}")
    print(f"Input seismic tensor shape: {mock_seismic.shape}")
    print(f"Output logits tensor shape: {logits.shape}")
    
    probabilities = torch.sigmoid(logits)
    print(f"Output probabilities:\n{probabilities}")
    print("Model check complete. Dimension mapping is correct!")
