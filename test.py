import torch
import matplotlib.pyplot as plt
import numpy as np
from model import model, denoise_process

def visualize_image(tensor_image):
    image = tensor_image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def generate_image(model, input_image, num_steps):
    model.eval()
    with torch.no_grad():
        denoised_image = denoise_process(model, input_image, num_steps)
        return denoised_image

model.load_state_dict(torch.load('Stable_Diffusion_2_epochs.pth'))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

input_image = torch.randn((1, 3, 32, 32))
input_image = input_image.to('cuda' if torch.cuda.is_available() else 'cpu')

num_steps = 20
generated_image = generate_image(model, input_image, num_steps)

visualize_image(generated_image.squeeze(0))
