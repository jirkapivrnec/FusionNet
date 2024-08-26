import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import datetime
import time

from FusionNet import FusionNet


# Step 1: Setup and Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"./log/{timestamp}_training_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)


batch_size = 64
learning_rate = 0.001
momentum = 0.9 # Used in SGD Optimizer
weight_decay=0.0005
epochs = 5

lr_strategy = "warmup" # none, warmup, oscilate, reset_lr
optim_algo = "SGD" # SGD, Adamax
model_to_load = None #"checkpoint_20240807_110214.pth"

# Log the device being used
logging.info(f"Using device: {device}, model: EMBRNet-CIFAR10, "
             f"epochs {epochs}, "
             f"fine-tune = {model_to_load}, "
             f"batch_size={batch_size}, "
             f"optim.{optim_algo}, lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}, "
             "mode='min', factor=0.1, patience=2, verbose=True, "
             "transforms=")

# Step 2: Load the FashionMNIST Dataset
dataset_path = '../data'

# Define data augmentation transforms
train_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    #transforms.RandomCrop(32, padding=4),  # Randomly crop images
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))  # Normalize the images
])

# No data augmentation for validation/testing, only normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5,0.5))
])


trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Instantiate and move the CNN model to GPU if available
model = FusionNet().to(device)

# Load model and optimizer state in case of continuing and/or fine tuning
if model_to_load:
    checkpoint = torch.load(f'./model/{model_to_load}')
    model.load_state_dict(checkpoint['model_state_dict'])

# Load model only for fine tuning
#model.load_state_dict(torch.load('./model/best_model_20240807_110214.pth'))  # Load the model to be fine tuned


# Step 4: Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # The lower lr the higher momentum should be. e.g. lr=0.1 mom=0.8 | lr=0.01 mom=0.9 | lr=0.001 mom=0.95
#optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# In case we want to continue in training on more epochs, load the optimizer
if model_to_load:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)



# Step 5: Train the Neural Network with Early Stopping and Validation Loss Tracking
patience = 10  # Early stopping patience
best_val_loss = float('inf')
trigger_times = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    epoch_start_time = time.time()  # Start time of the epoch

    # reset the lr for now
    if lr_strategy == "reset_lr":
        if epoch == 0:
            for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate 
    
    if lr_strategy == "warmup":
        # Starts with a small lr and increase it after some time to reduce it back
        if epoch > 0 and epoch <4:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 2 # Increase the lr from 0.001 -> 0.01
        
        elif epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate # Return back to the original lr
                
                if optim_algo == "SGD":
                    param_group['momentum'] = 0.95 #  Increase momentum


    # After some epochs, adjust lr and momentum
    """ if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
            param_group['momentum'] = 0.9
    
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
            param_group['momentum'] = 0.95 """

    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, epoch=epoch, train=True)  # Pass the current epoch to the forward function and a Train param
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
    
    val_loss = val_running_loss / len(testloader)
    val_losses.append(val_loss)

    # Update scheduler with the latest validation loss
    scheduler.step(val_loss)

    # Calculate the time taken for the epoch
    epoch_duration = time.time() - epoch_start_time  # Calculate the duration of the epoch

    
    
    log_message = (f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, "
                   f"Time: {epoch_duration:.2f} seconds, "
                   f"LR: {optimizer.param_groups[0]['lr']},")
                   #f", Branch 1 count: {branch1_count}, "
                   #f"Branch 2 count: {branch2_count}")
    logging.info(log_message)
    print(log_message)

    # Early Stopping Check
    """ if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break """




model_name = f"./model/best_model_{timestamp}.pth"
checkpoint_name = f"./model/checkpoint_{timestamp}.pth"

# Save model and optimizer state into a checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_name)


# Save only model
torch.save(model.state_dict(), model_name)  # Save the best model
logging.info('Finished Training')

# Step 7: Evaluate the Neural Network

model.load_state_dict(torch.load(model_name))  # Load the best model
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy}%')
logging.info(f'Accuracy of the network on the 10000 test images: {accuracy}%')


# Save the Training and Validation Losses plot
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')

loss_plot_filename = f"./log/{timestamp}_loss_plot.png"
plt.savefig(loss_plot_filename)
plt.close()

# After evaluating the model
# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testset.classes)
disp.plot(cmap=plt.cm.Blues)
cm_plot_filename = f"./log/{timestamp}_cm.png"
plt.savefig(cm_plot_filename)
plt.close()

# Error Analysis
misclassified_indices = np.where(np.array(all_labels) != np.array(all_preds))[0]
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(misclassified_indices[:5]):
    ax = axes[i]
    ax.imshow(testset.data[idx])
    ax.set_title(f"True: {all_labels[idx]}, Pred: {all_preds[idx]}")
    ax.axis('off')
ea_plot_filename = f"./log/{timestamp}_ea.png"
plt.savefig(ea_plot_filename)
plt.close()
