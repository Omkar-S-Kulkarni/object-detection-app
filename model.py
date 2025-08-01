# ======================================
# Pneumonia Detection with DenseNet121
# ======================================

#IMPORTS 
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from time import time as timer
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#  SET DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#  TRANSFORMS (Grayscale -> 3-channel RGB + Resize + Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#  DATA LOADING
train_dataset = datasets.ImageFolder("C:\\Users\\svks6\\Complete object detection\\data\\train", transform=transform)
test_dataset = datasets.ImageFolder("C:\\Users\\svks6\\Complete object detection\\data\\test", transform=transform)
val_dataset = datasets.ImageFolder("C:\\Users\\svks6\\Complete object detection\\data\\validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#  LOAD DENSENET121 MODEL
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)  # Binary classification
model = model.to(device)

#  LOSS AND OPTIMIZER
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#  TRAINING LOOP with EARLY STOPPING
train_losses = []
val_losses = []
early_stopping_patience = 5
no_improve_count = 0
best_val_loss = float('inf')
start_time = timer()

for epoch in tqdm(range(30)):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Match shape for BCE
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        no_improve_count += 1
        if no_improve_count >= early_stopping_patience:
            print("Early stopping triggered.")
            break

#  LOAD BEST MODEL
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# EVALUATE ON TEST SET
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).int().squeeze(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#  ACCURACY
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#  CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#LOSS CURVE PLOT
def plot_loss_curve(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_loss_curve(train_losses, val_losses)

# PREDICTION VISUALIZATION
import random

def visualize_predictions(model, dataloader, class_names):
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            predicted = (preds > 0.5).int().squeeze(1)
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i in range(5):
                idx = random.randint(0, len(images)-1)
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                label = class_names[labels[idx]]
                pred = class_names[predicted[idx]]
                axes[i].imshow(img)
                axes[i].set_title(f"Label: {label}\nPred: {pred}")
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
            break  # Show only one batch

visualize_predictions(model, test_loader, train_dataset.classes)

# TOTAL TRAINING TIME
end_time = timer()
print(f"Total training time: {end_time - start_time:.2f} seconds")
