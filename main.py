import torch
import torch.nn as nn
import torch.optim as optim
from model import model, criterion, optimizer, denoise_process
from build_dataset import train_loader

num_steps = 10
learning_rate = 0.001
num_epochs = 2

total_batches = len(train_loader)

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        x = data
        noise = torch.randn_like(x)
        noisy_image = x + noise

        denoised_image = denoise_process(model, noisy_image, num_steps)

        loss = criterion(denoised_image, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            progress = (i + 1) / total_batches * 100
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_batches}], "
                  f"Progress: {progress:.2f}%, Loss: {loss.item()}")

torch.save(model.state_dict(), 'Stable_Diffusion_2_epochs.pth')

print("Training Complete.")
