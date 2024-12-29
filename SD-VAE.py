# %%
import wandb
import time
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
# from tqdm.notebook import tqdm, trange

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator, notebook_launcher
import clip
import ipywidgets as widgets
import dotenv
import os

dotenv.load_dotenv()

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
        self.final_layer_norm = nn.LayerNorm((out_channels, output_width, output_width))
    
    def forward(self, x):
        assert(self.out_channels // self.in_channels in [2, 4])
        space_2_channel_x = space_2_channel(x, self.out_channels // self.in_channels == 4)

        for index, (conv_layer, layer_norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            y = conv_layer(x)
            y = layer_norm(y)
            if index < len(self.conv_layers) - 1:
                x = self.activation(x + y)
            else:
                x = self.activation(space_2_channel_x + y)
        return self.final_layer_norm(x)

class ResConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_width, kernel_size, stride, padding, activation=nn.Tanh(), depth=1):
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
        self.final_layer_norm = nn.LayerNorm((out_channels, output_width, output_width))
    
    def forward(self, x):
        assert(self.in_channels // self.out_channels in [2, 4])
        channel_2_space_x = channel_2_space(x)
        for index, (conv_layer, layer_norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            y = conv_layer(x)
            y = layer_norm(y)
            if index < len(self.conv_layers) - 1:
                x = self.activation(x + y)
            else:
                x = self.activation(channel_2_space_x + y)
        return self.final_layer_norm(x)

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
        self.lin_dims = [96 * 8 * 8, 96 * 8, 32 * 8, 128, 64, 32, 16, 1]
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
        for index, lin_layer in enumerate(self.lin_layers):
            x = lin_layer(x)
            if index < len(self.lin_layers) - 1:
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

    def get_encoder_output(self, x):
        return self.encoder(x)

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
        det_diff = torch.mean(logvar2 - logvar1)
        ratio = torch.mean(torch.exp(logvar1 - logvar2))
        mean_diff = torch.mean(((mu1 - mu2).pow(2) / torch.exp(logvar2)))
        return 0.5 * (det_diff + ratio + mean_diff)
    
    def train_step(self, batch, optimizer, disc_optimizer, scheduler, clip_model, resize, accelerator=None, enable_wandb=False, teach_gan=False, enable_gan=False, gan_coeff=0.0, beta_viz=1.0, beta_kl=0.001, beta_l2=1.0):
        is_main_process = accelerator is not None and accelerator.is_main_process
        encoded_info = self.get_encoder_output(batch)
        mu, logvar = self.split_code(encoded_info)
        eps = torch.randn_like(mu)
        encoded = mu + eps * torch.exp(logvar)
        output = self.decode(encoded)

        kl_loss = self.KL_loss(mu, logvar, torch.zeros_like(mu), torch.zeros_like(mu))
        # print(f"KL loss: {kl_loss}")
        viz_loss = CLIP_loss(batch, output, clip_model, resize)
        # print(f"Viz loss: {viz_loss}")
        mse = nn.MSELoss()
        l2_loss = mse(batch, output)
        # print(f"L2 loss: {l2_loss}")

        loss = beta_viz * viz_loss + beta_kl * kl_loss + beta_l2 * l2_loss
        if teach_gan:
            gan_fake_loss = -((torch.sigmoid(self.discriminator(output))+1e-8).log()).mean()
            gan_real_loss = -((1-torch.sigmoid(self.discriminator(batch)) + 1e-8).log()).mean()
            gan_loss = gan_fake_loss + gan_real_loss
            # print(f"GAN loss: {gan_loss}")
            disc_optimizer.zero_grad()
            gan_loss.backward(retain_graph=True)
        
        if enable_gan and teach_gan:
            loss -= gan_coeff * gan_loss

        optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if teach_gan:
            disc_optimizer.step()

        gathered_loss = accelerator.gather(loss)
        gathered_viz_loss = accelerator.gather(viz_loss)
        gathered_kl_loss = accelerator.gather(kl_loss)
        gathered_l2_loss = accelerator.gather(l2_loss)

        if enable_wandb and is_main_process:
            wandb.log({"loss": gathered_loss.mean().item()})
            wandb.log({"viz_loss": gathered_viz_loss.mean().item()})
            wandb.log({"kl_loss": gathered_kl_loss.mean().item()})
            wandb.log({"l2_loss": gathered_l2_loss.mean().item()})

        if teach_gan:
            gathered_gan_loss = accelerator.gather(gan_loss)
            if is_main_process:
                wandb.log({"gan_loss": gathered_gan_loss.mean().item()})

        return loss.item()

    def train(self, dataloader, optimizer, disc_optimizer, scheduler, clip_model, resize, epochs=100, accelerator=None, enable_wandb=False, enable_gan=False, enable_gan_cutoff=0.8, beta_viz=1.0, beta_kl=0.001, beta_l2=1.0):
        best_loss = float('inf')
        # Get process info from accelerator
        is_main_process = accelerator is not None and accelerator.is_main_process
        progress_bar = tqdm(range(epochs), 
                           desc=f"Training (process {accelerator.process_index})", 
                           position=accelerator.process_index*2,
                           leave=True,
                           disable=not is_main_process)
        batch_progress = tqdm(total=len(dataloader), 
                             desc=f"Batches (process {accelerator.process_index})", 
                             position=accelerator.process_index*2+1,
                             leave=True,
                             disable=not is_main_process)
        
        num_steps = len(dataloader) * epochs
        gan_coeffs = iter(np.linspace(0, 1, num_steps) ** 2)
        for _ in progress_bar:
            epoch_loss = 0
            batch_progress.reset()
            
            for index, batch in enumerate(dataloader):
                batch = batch.to(next(self.parameters()).device)

                now_enable_gan = enable_gan and (index / len(dataloader) > enable_gan_cutoff)
                loss = self.train_step(batch, optimizer, disc_optimizer, scheduler, clip_model, resize, accelerator, enable_wandb, teach_gan=enable_gan, enable_gan=True, gan_coeff=next(gan_coeffs), beta_viz=beta_viz, beta_kl=beta_kl, beta_l2=beta_l2)
                epoch_loss += loss
                batch_progress.update(1)
                if index % 20 == 0:
                    self.visual_eval(accelerator, index, enable_wandb)
                
            epoch_loss /= len(dataloader)
            progress_bar.set_postfix({'loss': f'{epoch_loss:.4f}'})
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if is_main_process:
                    torch.save(self.state_dict(), "long_sana_vae.pt")
        
        # if is_main_process:
            # print(f"Best loss: {best_loss}")
    
    def visual_eval(self, accelerator=None, index=0, enable_wandb=False):
        is_main_process = accelerator is not None and accelerator.is_main_process
        is_main_process = accelerator is None or is_main_process
        if not is_main_process:
            return

        # Load and transform a single image for visualization
        img_path = f"/shared/imagenet/train/image_0.jpg"
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(next(self.parameters()).device)

        # Pass through model
        with torch.no_grad():
            recon = self(img_tensor)
            
        # Convert output tensor to PIL image
        recon_img = transforms.ToPILImage()(recon.squeeze().cpu())
        recon_img.save(f"test_imgs/reconstructed_image_0_{index}.jpg")
        # Save reconstructed image to wandb if enabled
        # if is_main_process:
        #     print(f"Eval'd reconstructed image to `test_imgs/reconstructed_image_0_{index}.jpg`")
        if enable_wandb and is_main_process:
            wandb.log({
                "original_image": wandb.Image(img),
                "reconstructed_image": wandb.Image(recon_img)
            })

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

glob_wandb_on = True
enable_gan = True

def special_print(my_str, accelerator, start_time):
    if accelerator.is_main_process:
        print(my_str)
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"Elapsed time: {minutes:02d}:{seconds:02d}")
# %%
def training_function():
    # Make accelerator global so it's accessible in the train method
    start_time = time.time()
    accelerator = Accelerator()
    special_print("Accelerator initialized", accelerator, start_time)
    beta_viz = 1.0
    beta_kl = 0.0010
    beta_l2 = 1.0
    enable_gan_cutoff = 0.7
    enable_gan = True
    lr = 1e-3
    disc_lr = 1e-4
    epochs = 24

    is_main_process = accelerator.is_main_process
    if is_main_process and glob_wandb_on:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="monoscripts-vae", config={"lr": lr, "beta_viz": beta_viz, "beta_kl": beta_kl, "beta_l2": beta_l2, "disc_lr": disc_lr, "enable_gan_cutoff": enable_gan_cutoff, "epochs": epochs, "enable_gan": enable_gan, "super_compile": True})
        special_print("Wandb initialized", accelerator, start_time)

    dataloader = get_dataloader(batch_size=256)
    special_print("Loaded dataloader", accelerator, start_time)

    clip_model, resize = get_clip_model(device=accelerator.device)

    model = SANA_VAE()
    special_print(f"Number of parameters: {num_params(model)}", accelerator, start_time)
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3, betas=(0.9, 0.99))
    disc_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

    real_ds_len = len(dataloader) * epochs // 8
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=real_ds_len // 8),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=real_ds_len * 7 // 8)
    ])

    model, optimizer, disc_optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, disc_optimizer, scheduler, dataloader)
    model = torch.compile(model)
    if is_main_process:
        special_print("Prepared and compiled model", accelerator, start_time)
        if glob_wandb_on:
            wandb.watch(model)

    SANA_VAE.train(model.module if hasattr(model, 'module') else model, dataloader, optimizer, disc_optimizer, scheduler, clip_model, resize, epochs=epochs, accelerator=accelerator, enable_wandb=glob_wandb_on, enable_gan=enable_gan, enable_gan_cutoff=enable_gan_cutoff, beta_viz=beta_viz, beta_kl=beta_kl, beta_l2=beta_l2)

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
if __name__ == "__main__":
    training_function()