# %%
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm, trange

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator, notebook_launcher
import clip

# %%
# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn', force=True)

# %%
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all image files
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def undo_transform(x):
    return x * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])

# Create dataset and dataloader
def get_dataloader(batch_size=256):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNetDataset(root_dir='/shared/imagenet/train', 
                            transform=transform)
    dataloader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)
    return dataloader

# %%
def get_clip_model(device='cpu'):
    if device == 'cpu':
        clip_model, preprocess = clip.load("ViT-B/32")
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    resize = transforms.Resize((224, 224))
    return clip_model, resize

def CLIP_loss(truth_batch, output_batch, clip_model, resize, device=None):
    if device is None:
        device = truth_batch.device
    truth_scores = clip_model.encode_image(resize(truth_batch).to(device))
    output_scores = clip_model.encode_image(resize(output_batch).to(device))
    return (truth_scores - output_scores).pow(2).mean()

# %%
def space_2_channel(x, no_compression=False):
    if no_compression:
        output = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2], x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        assert(output.shape == (x.shape[0], x.shape[1] * 4, x.shape[2] // 2, x.shape[3] // 2))
    else:
        output = torch.cat((x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2], x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]), dim=1) / 2
        assert(output.shape == (x.shape[0], x.shape[1] * 2, x.shape[2] // 2, x.shape[3] // 2))
    return output

def channel_2_space(x):
    output = torch.zeros((x.shape[0], x.shape[1] // 2, x.shape[2] * 2, x.shape[3] * 2), device=x.device)
    half_C = x.shape[1] // 2
    output[:, :, ::2, ::2] = x[:, :half_C, :, :]
    output[:, :, ::2, 1::2] = x[:, half_C:, :, :]
    output[:, :, 1::2, ::2] = x[:, half_C:, :, :]
    output[:, :, 1::2, 1::2] = x[:, :half_C, :, :]
    return output

kernel_sizes = [11, 9, 5, 3, 3]
strides = [2, 2, 2, 2, 2]
paddings = [k//2 for k in kernel_sizes]

# %%
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_width, kernel_size, stride, padding, activation=nn.LeakyReLU(0.2), depth=1):
        super().__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_width = output_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv_layers = []
        layer_norms = []

        for index in range(depth):
            if index < depth-1:
                conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding))
                layer_norms.append(nn.LayerNorm((in_channels, 2 * output_width, 2 * output_width)))
            else:
                conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
                layer_norms.append(nn.LayerNorm((out_channels, output_width, output_width)))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norms = nn.ModuleList(layer_norms)
    
    def forward(self, x):
        assert(self.out_channels // self.in_channels in [2, 4])
        space_2_channel_x = space_2_channel(x, self.out_channels // self.in_channels == 4)

        for index, (conv_layer, layer_norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            y = conv_layer(x)
            y = self.activation(y)
            if index < len(self.conv_layers) - 1:
                x = x + layer_norm(y)
            else:
                x = space_2_channel_x + y
        return x

class ResConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_width, kernel_size, stride, padding, activation=nn.LeakyReLU(0.2), depth=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_width = output_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        conv_layers = []
        layer_norms = []
        for index in range(depth):
            if index < depth-1:
                conv_layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=1, padding=padding))
                layer_norms.append(nn.LayerNorm((in_channels, output_width // 2, output_width // 2)))
            else:
                conv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=stride-1))
                layer_norms.append(nn.LayerNorm((out_channels, output_width, output_width)))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norms = nn.ModuleList(layer_norms)
    
    def forward(self, x):
        assert(self.in_channels // self.out_channels in [2, 4])
        channel_2_space_x = channel_2_space(x)
        for index, (conv_layer, layer_norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            y = conv_layer(x)
            y = self.activation(y)
            if index < len(self.conv_layers) - 1:
                x = x + layer_norm(y)
            else:
                x = channel_2_space_x + y
        return x

# %%
class SANA_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [3, 12, 24, 48, 96, 192]
        self.output_widths = [256, 128, 64, 32, 16, 8]
        self.depths = [40, 20, 10, 10, 10]
        self.activation = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(
            *[ResConvBlock(self.channels[index], self.channels[index+1], self.output_widths[index+1], kernel_size, stride, padding, activation=self.activation, depth=self.depths[index])
              for index, kernel_size, stride, padding in zip(range(len(kernel_sizes)), kernel_sizes, strides, paddings)]
        )

    def forward(self, x):
        return self.layers(x)

class SANA_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [3, 6, 12, 24, 48, 96]
        self.output_widths = [256, 128, 64, 32, 16, 8]
        self.depths = [40, 20, 10, 10, 10, 10]
        self.activation = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(
            *[ResConvTransposeBlock(self.channels[index+1], self.channels[index], self.output_widths[index], kernel_size, stride, padding, activation=self.activation, depth=self.depths[index])
              for index, kernel_size, stride, padding in zip(range(len(kernel_sizes)), kernel_sizes, strides, paddings)][::-1]
        )

    def forward(self, x):
        return self.layers(x)

# %%
class SANA_Discriminator(nn.Module):
    # returns logit of whether the image is real
    def __init__(self):
        super().__init__()
        self.channels = [3, 6, 12, 24, 48, 96]
        self.kernel_sizes = [11, 9, 5, 3, 3]
        self.strides = [2, 2, 2, 2, 2]
        self.paddings = [k//2 for k in self.kernel_sizes]
        self.lin_dims = [96 * 16 * 16, 96 * 8, 32 * 8, 128, 64, 32, 16, 1]
        self.activation = nn.LeakyReLU(0.2)
        self.convlayers = nn.ModuleList([
            nn.Conv2d(self.channels[index], self.channels[index+1], kernel_size, stride, padding)
            for index, kernel_size, stride, padding in zip(range(len(self.channels)-1), self.kernel_sizes, self.strides, self.paddings)
        ])
        self.lin_layers = nn.ModuleList([
            nn.Linear(self.lin_dims[index], self.lin_dims[index+1]) for index in range(len(self.lin_dims)-1)
        ])
    
    def forward(self, x):
        for conv_layer in self.convlayers:
            x = conv_layer(x)
            x = self.activation(x)
        x = x.view(x.shape[0], -1)
        for lin_layer in self.lin_layers:
            x = lin_layer(x)
            x = self.activation(x)
        return x

# %%
"""
SD-VAE

Replicates f8c4p2 on 256x256 ImageNet, i.e. 256x256 -> 16x16x12.

Conv structure:
* 
"""

class SANA_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SANA_Encoder()
        self.decoder = SANA_Decoder()
        self.discriminator = SANA_Discriminator()
    
    def split_code(self, x):
        return x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]

    def encode(self, x, eps=None):
        output = self.encoder(x)
        mu, logvar = self.split_code(output)
        if eps is None:
            eps = torch.randn_like(mu).to(mu.device)
        return mu + eps * torch.exp(logvar)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, eps=None):
        return self.decode(self.encode(x, eps))
    
    def KL_loss(self, mu1, logvar1, mu2, logvar2):
        det_diff = torch.sum(logvar1 - logvar2)
        ratio = torch.exp(logvar1 - logvar2).sum()
        mean_diff = ((mu1 - mu2).pow(2) / torch.exp(logvar2)).sum()
        return 0.5 * (det_diff + ratio + mean_diff)
    
    def train_step(self, batch, optimizer, clip_model, resize):
        optimizer.zero_grad()
        encoded_info = self.encoder(batch)
        mu, logvar = self.split_code(encoded_info)
        eps = torch.randn_like(mu)
        encoded = mu + eps * torch.exp(logvar)
        output = self.decode(encoded)

        kl_loss = self.KL_loss(mu, logvar, torch.zeros_like(mu), torch.ones_like(mu))
        viz_loss = CLIP_loss(batch, output, clip_model, resize)
        mse = nn.MSELoss()
        l2_loss = mse(batch, output)
        gan_loss = 0

        loss = viz_loss + kl_loss + l2_loss + gan_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, dataloader, optimizer, clip_model, resize, epochs=100):
        best_loss = float('inf')
        for _ in trange(epochs, desc="Epochs", leave=True):
            epoch_loss = 0
            for batch in tqdm(dataloader, desc="Batches", leave=True):
                batch = batch.to(next(self.parameters()).device)
                loss = SANA_VAE.train_step(self, batch, optimizer, clip_model, resize)
                epoch_loss += loss
            epoch_loss /= len(dataloader)
            print(f"Loss: {epoch_loss}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.state_dict(), "sana_vae.pt")
        print(f"Best loss: {best_loss}")

# %%
def num_params(model):
    return sum(p.numel() for p in model.parameters())

# %%
# print("VAE params:", num_params(SANA_VAE()))
# print("Discriminator params:", num_params(SANA_Discriminator()))
# print("Decoder params:", num_params(SANA_Decoder()))
# print("Encoder params:", num_params(SANA_Encoder()))

# %%
# clip_model, resize = get_clip_model('cuda')
# model = SANA_VAE().to('cuda')
# model = torch.compile(model)
# print(f"Number of parameters: {num_params(model)}")
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# SANA_VAE.train(model, dataloader, optimizer, clip_model, resize, epochs=10)

# %%
def training_function():
    accelerator = Accelerator()

    dataloader = get_dataloader(batch_size=256)

    clip_model, resize = get_clip_model(device=accelerator.device)

    model = SANA_VAE()
    print(f"Number of parameters: {num_params(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model = torch.compile(model)

    SANA_VAE.train(model, dataloader, optimizer, clip_model, resize, epochs=10)

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
if __name__ == "__main__":
    training_function()