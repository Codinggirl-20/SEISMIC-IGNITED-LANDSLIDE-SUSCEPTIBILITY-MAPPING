import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import our custom modules
from dataset import LandslideSusceptibilityDataset, generate_synthetic_data
from model import HybridLandslideModel

# Try importing sklearn and matplotlib for metrics and plotting.
# If they aren't available, we'll log fallback messages so execution doesn't crash.
try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def train_model(epochs=20, batch_size=32, lr=0.001):
    print("==================================================")
    print("   Starting Hybrid Landslide DL Model Training    ")
    print("==================================================")
    
    # 1. Generate synthetic geographic-seismic dataset
    print("[1/5] Generating multi-modal synthetic dataset...")
    spatial_data, seismic_data, labels = generate_synthetic_data(num_samples=1200, patch_size=15)
    
    # Create Dataset instance
    dataset = LandslideSusceptibilityDataset(spatial_data, seismic_data, labels)
    
    # Train/Validation Split (80% Train, 20% Val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Fix seed for split reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset generated. Train samples: {train_size}, Validation samples: {val_size}")
    
    # 2. Instantiate Model, Loss, and Optimizer
    print("[2/5] Initializing Hybrid PyTorch Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")
    
    model = HybridLandslideModel(spatial_channels=6, seismic_dim=3, patch_size=15).to(device)
    
    # Binary Cross Entropy with Logits combines sigmoid and cross-entropy for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Track metrics for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # 3. Training Loop
    print("[3/5] Starting epochs training loops...")
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for spatial_patches, seismic_feats, batch_labels in train_loader:
            spatial_patches = spatial_patches.to(device)
            seismic_feats = seismic_feats.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1) # Reshape to (Batch, 1)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(spatial_patches, seismic_feats)
            loss = criterion(logits, batch_labels)
            
            # Backward pass & step
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * spatial_patches.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for spatial_patches, seismic_feats, batch_labels in val_loader:
                spatial_patches = spatial_patches.to(device)
                seismic_feats = seismic_feats.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                
                logits = model(spatial_patches, seismic_feats)
                loss = criterion(logits, batch_labels)
                val_running_loss += loss.item() * spatial_patches.size(0)
                
                # Calculate probability and predicted class
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                correct_predictions += (preds == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)
                
                # Store for ROC-AUC
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions / total_predictions
        
        # Update learning rate scheduler
        scheduler.step(val_epoch_loss)
        
        # Compute AUC and F1 if sklearn is available
        auc_score = 0.5
        f1 = 0.0
        if SKLEARN_AVAILABLE:
            auc_score = roc_auc_score(all_labels, all_probs)
            f1 = f1_score(all_labels, [1 if p >= 0.5 else 0 for p in all_probs])
            
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_accuracy)
        history['val_auc'].append(auc_score)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2%} | "
              f"Val AUC: {auc_score:.4f} | "
              f"Val F1: {f1:.4f}")
        
        # Save model if validation loss improves
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/hybrid_landslide_model.pth")
            
    print("\n[4/5] Training finished. Model weights saved to 'models/hybrid_landslide_model.pth'")
    
    # 4. Save validation targets and predictions for visual analysis
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # 5. Generate validation plots
    if MATPLOTLIB_AVAILABLE and SKLEARN_AVAILABLE:
        print("[5/5] Generating evaluation metrics visualizations...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss Curves
        axes[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#1a73e8', linewidth=2)
        axes[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#d93025', linewidth=2)
        axes[0].set_title('Model Convergence (Cross Entropy Loss)', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss Value')
        axes[0].legend(frameon=True)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        axes[1].plot(fpr, tpr, color='#137333', linewidth=2.5, label=f'Hybrid Model (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], color='#5f6368', linestyle='--', label='Random Susceptibility (AUC = 0.500)')
        axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
        axes[1].set_ylabel('True Positive Rate (Sensitivity)')
        axes[1].legend(loc='lower right', frameon=True)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plot_path = "outputs/model_evaluation.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Performance plots successfully generated and saved to '{plot_path}'")
        plt.close()
    else:
        print("[5/5] Matplotlib or scikit-learn is not installed. Skipping evaluation plotting.")
        print("Install dependencies using 'pip install -r requirements.txt' to generate plots.")

if __name__ == "__main__":
    train_model(epochs=20, batch_size=32, lr=0.001)
