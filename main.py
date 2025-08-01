# =============================================
# main.py (Place this in root directory)
# =============================================
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torch import nn
import time
import os
from collections import Counter
import numpy as np
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

from src.model import build_densenet_model
from src.train import train
from src.test import evaluate, visualize_predictions
from src.utils import get_transforms, plot_loss_curve

def create_balanced_sampler(dataset):
   
    if hasattr(dataset, 'imgs'):
        labels = [label for _, label in dataset.imgs]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    class_count = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in class_count.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def main():
    
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Check data directory
    if not os.path.exists("data"):
        print("\n Error: 'data' directory not found!")
        print("Please organize your data as follows:")
        print("data/")
        print(" train/")
        print(" NORMAL/")
        print(" PNEUMONIA/")
        print(" validation/")
        print(" NORMAL/")
        print(" PNEUMONIA/")
        print(" test/")
        print("   NORMAL/")
        print("  PNEUMONIA/")
        return
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load datasets
    print("\n Loading datasets...")
    train_transform = get_transforms('train')
    test_transform = get_transforms('test')
    
    try:
        train_data = datasets.ImageFolder("data/train", transform=train_transform)
        val_data = datasets.ImageFolder("data/validation", transform=test_transform)
        test_data = datasets.ImageFolder("data/test", transform=test_transform)
    except FileNotFoundError as e:
        print(f" Error loading data: {e}")
        return
    
    # Print dataset info
    print(f"\n Dataset Information:")
    print(f"Training samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Classes: {train_data.classes}")
    
    # Check class distribution
    if hasattr(train_data, 'imgs'):
        labels = [label for _, label in train_data.imgs]
    else:
        labels = [train_data[i][1] for i in range(len(train_data))]
    
    class_distribution = Counter(labels)
    print(f"Class Distribution: {dict(class_distribution)}")
    
    # Calculate class imbalance ratio
    normal_count = class_distribution[0]
    pneumonia_count = class_distribution[1]
    imbalance_ratio = max(normal_count, pneumonia_count) / min(normal_count, pneumonia_count)
    print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Create data loaders with balanced sampling
    batch_size = 16  # Reduced batch size to prevent overfitting
    
    # Use balanced sampler for training
    train_sampler = create_balanced_sampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Build model with proper regularization
    print(f"\n  Building model...")
    model = build_densenet_model(num_classes=1, dropout_rate=0.5).to(device)
    
    # Use class weights in loss function to handle imbalance
    pos_weight = torch.tensor([normal_count / pneumonia_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate
        weight_decay=1e-3,  # L2 regularization
        betas=(0.9, 0.999)
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters()) - trainable_params:,}")
    
    # Training
    print(f"\n Starting training...")
    start_time = time.time()
    
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n  Total training time: {training_time/60:.1f} minutes")
    
    # Plot results
    print(f"\n Generating plots...")
    plot_loss_curve(train_losses, val_losses)
    
    # Define the path to the best model
    best_model_path = "outputs/best_model.pth"
    
    # Load the best model if it exists
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(" Successfully loaded best model!")
        except Exception as e:
            print(f" Could not load best model: {e}")
            print("Using current model state for evaluation")
    else:
        print(" Best model file not found, using current model state")
    
    print(f"\ Evaluating on test set...")
    test_accuracy, roc_auc = evaluate(model, test_loader, device, class_names=train_data.classes)
    
    print(f"\  Generating prediction visualizations...")
    visualize_predictions(model, test_loader, device)
    
    # Final summary
    print(f"\n" + "="*60)
    print(f" TRAINING COMPLETED SUCCESSFULLY!")
    print(f"="*60)
    print(f" Final Results:")
    print(f"   • Test Accuracy: {test_accuracy:.2f}%")
    print(f"   • ROC AUC Score: {roc_auc:.3f}")
    print(f"   • Training Time: {training_time/60:.1f} minutes")
    print(f"   • Best Model: {best_model_path}")
    print(f" Generated Files:")
    print(f"   • {best_model_path}")
    print(f"   • outputs/loss_curve.png")
    print(f"   • outputs/confusion_matrix.png")
    print(f"   • outputs/roc_curve.png")
    print(f"   • outputs/predictions_visualization.png")
    
    print(f"\ Ready for Streamlit app!")
    print(f"Run: streamlit run app.py")

if __name__ == "__main__":
    main()