# %%
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, Optional, List
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm
from typing import Callable
from jaxtyping import Float, Int
import einops
from dataclasses import dataclass
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DistributedDataParallelKwargs, notebook_launcher
from torch.profiler import profile, record_function, ProfilerActivity
import ipywidgets as widgets

# %%
import dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
import os
dotenv.load_dotenv()
import huggingface_hub
HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
huggingface_hub.login(token=HUGGINGFACE_API_KEY)

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("bigcode/the-stack-v2", cache_dir="/shared/alex-zhao-storage/the-stack-v2", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/shared/alex-zhao-storage/hf-cache")

# %%
tokenizer.vocab_size

# %%
# device = torch.device("cuda")

# a utility for calculating running average
class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val: float, sz: float):
        self.num += val*sz
        self.tot += sz

    def calculate(self) -> float:
        return self.num/self.tot

# %% [markdown]
# # Prefix Ops

# %%
from math import log2
class PrefixOps():
    # Assumes that position is a power of 2
    def pref_mul(t: Float[Tensor, "batch position d_model d_state"]):
        n_layers = int(log2(t.shape[1]))
        up_tensors = []
        down_tensors = []
        up_tensors.append(t)
        for _ in range(n_layers):
            left = up_tensors[-1][:, ::2]
            right = up_tensors[-1][:, 1::2]
            up_tensors.append(left * right)

        down_tensors.append(torch.ones_like(up_tensors[-1]))
        for index in range(n_layers):
            new = torch.zeros_like(up_tensors[-2 - index])
            new[:, ::2] = down_tensors[-1]
            new[:, 1::2] = down_tensors[-1] * up_tensors[-2 - index][:, ::2]
            down_tensors.append(new)
        return down_tensors[-1] * t

    def pref_add(t: Float[Tensor, "batch position d_model d_state"]):
        n_layers = int(log2(t.shape[1]))
        up_tensors = []
        down_tensors = []
        up_tensors.append(t)
        for _ in range(n_layers):
            left = up_tensors[-1][:, ::2]
            right = up_tensors[-1][:, 1::2]
            up_tensors.append(left + right)

        down_tensors.append(torch.zeros_like(up_tensors[-1]))
        for index in range(n_layers):
            new = torch.zeros_like(up_tensors[-2 - index])
            new[:, ::2] = down_tensors[-1]
            new[:, 1::2] = down_tensors[-1] + up_tensors[-2 - index][:, ::2]
            down_tensors.append(new)
        return down_tensors[-1] + t

class TestPrefixOps():
    def __init__(self, batch, position, d_model, d_state):
        self.batch = batch
        self.position = position
        self.d_model = d_model
        self.d_state = d_state

    def pref_mul(self):
        my_in = torch.exp(torch.randn(self.batch, self.position, self.d_model, self.d_state))
        my_out = PrefixOps.pref_mul(my_in)
        true_out = torch.ones_like(my_in)
        for i in range(self.position):
            if i==0:
                true_out[:, i] = my_in[:, i]
            else:
                true_out[:, i] = true_out[:, i-1] * my_in[:, i]
        assert(torch.allclose(my_out, true_out))
    
    def pref_add(self):
        my_in = torch.randn(self.batch, self.position, self.d_model, self.d_state)
        my_out = PrefixOps.pref_add(my_in)
        true_out = torch.zeros_like(my_in)
        for i in range(self.position):
            if i==0:
                true_out[:, i] = my_in[:, i]
            else:
                true_out[:, i] = true_out[:, i-1] + my_in[:, i]
        assert(torch.allclose(my_out, true_out, atol=1e-4))

TestPrefixOps(12, 64, 768, 64).pref_mul()
TestPrefixOps(12, 64, 768, 64).pref_add()

# %% [markdown]
# # Implement Mamba

# %% [markdown]
# ## Single Head

# %%
# TODO: generate not supported for pref_sum
test_ssm_ablation = False
pref_sum = True
use_double = False
careful_double = True

class SSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()

        # get A by negative softplus
        self.Araw = nn.Parameter(torch.randn(d_model, d_state))
        torch.nn.init.kaiming_normal_(self.Araw)
        self.Araw.data = self.Araw.data.clamp(min=-4)
        self.s_B = nn.Linear(d_model, d_state)
        self.s_C = nn.Linear(d_model, d_state)
        self.s_D = nn.Linear(d_model, 1)
        self.P = nn.Parameter(torch.randn(d_model))
        # torch.nn.init.xavier_normal_(self.P)

        self.d_model = d_model
        self.d_state = d_state

    def forward(self, x: Float[Tensor, "batch position d_model"], keep_hidden=False, use_hidden=False) -> Float[Tensor, "batch position d_model"]:
        sp = nn.Softplus()
        disc = sp(self.P + self.s_D(x).repeat(1, 1, self.d_model)) # size batch position d_model
        A = -1 * sp(self.Araw)
        A_bar_pre = disc[..., None] * A # size batch position d_model d_state
        # A_bar_pre = torch.where(torch.abs(A_bar_pre) < 1e-4, -1e-4 * torch.ones_like(A_bar_pre), A_bar_pre)
        A_bar = torch.exp(A_bar_pre)
        B = self.s_B(x) # size batch position d_state
        # ratio = 1 + A_bar_pre/2 + A_bar_pre**2/2 + A_bar_pre**3/6 + A_bar_pre**4/24 + A_bar_pre**5/120
        B_bar = B[:, :, None] / A * (A_bar - 1)

        C = self.s_C(x)

        assert(str(x.device)[:4] == 'cuda')

        if pref_sum:
            Bx = B_bar * x[..., None]
            Bx_log = torch.log(torch.abs(Bx))
            A_bar_prod_log = PrefixOps.pref_add(A_bar_pre)
            # AinvsB_log must be at most 77
            if use_double:
                A_bar_prod_log = torch.max(A_bar_prod_log, Bx_log-150)
                if careful_double:
                    A_bar_prod_log = torch.max(A_bar_prod_log, Bx_log-70)
                AinvsB_log = Bx_log - A_bar_prod_log
                AinvsBsum = PrefixOps.pref_add(torch.exp(AinvsB_log.to(torch.double)) * torch.sign(Bx))
                assert(AinvsBsum.dtype == torch.double)
                y = torch.matmul((AinvsBsum * torch.exp(A_bar_prod_log.to(torch.double))).to(torch.float), C[..., None]).squeeze(-1).to(x.dtype)
            else:
                A_bar_prod_log = torch.max(A_bar_prod_log, Bx_log-70)
                AinvsB_log = Bx_log - A_bar_prod_log
                AinvsBsum = PrefixOps.pref_add(torch.exp(AinvsB_log) * torch.sign(Bx))
                y = torch.matmul((AinvsBsum * torch.exp(A_bar_prod_log)), C[..., None]).squeeze(-1)
            if torch.isnan(y).any() or torch.isinf(y).any():
                raise ValueError("NaN or Inf values detected in SSM output")
        else:
            if use_hidden:
                h = self.h
            else:
                h = torch.zeros(x.shape[0], self.d_model, self.d_state).to(x.device)
            y = torch.zeros_like(x).to(x.device)
            for index in range(x.shape[1]):
                if index == 0:
                    h = B_bar[:, index] * x[:, index].view(-1, self.d_model, 1)
                else:
                    h = A_bar[:, index] * h + B_bar[:, index] * x[:, index].view(-1, self.d_model, 1)
                y[:, index] = torch.matmul(h, C[:, index, :, None]).squeeze(-1)

        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("NaN or Inf values detected in SSM output")
        if keep_hidden:
            self.h = h
        return y

    def device(self):
        return next(self.parameters()).device
    
    def test_forward(self, x: Float[Tensor, "batch position d_model"]):
        sp = nn.Softplus()
        disc = sp(self.P + self.s_D(x).repeat(1, 1, self.d_model)) # size batch position d_model
        A_bar_pre = disc[..., None] * (-1 * sp(self.Araw)) # size batch position d_model d_state
        # A_bar_pre = torch.where(torch.abs(A_bar_pre) < 1e-4, -1e-4 * torch.ones_like(A_bar_pre), A_bar_pre)
        A_bar = torch.exp(A_bar_pre)
        B = self.s_B(x) # size batch position d_state
        ratio = 1 + A_bar_pre/2 + A_bar_pre**2/2 + A_bar_pre**3/6 + A_bar_pre**4/24 + A_bar_pre**5/120
        B_bar = ratio * (torch.unsqueeze(disc, -1) * torch.unsqueeze(B, 2)) # size batch position d_model d_state

        C = self.s_C(x)

        assert(str(x.device)[:4] == 'cuda')
        h = torch.zeros(x.shape[0], self.d_model, self.d_state).to(x.device)

        test_y = self.forward(x)

        y = torch.zeros_like(x).to(x.device)
        for index in range(x.shape[1]):
            if index == 0:
                h = B_bar[:, index] * x[:, index].view(-1, self.d_model, 1)
            else:
                h = A_bar[:, index] * h + B_bar[:, index] * x[:, index].view(-1, self.d_model, 1)
            y[:, index] = torch.matmul(h, C[:, index, :, None]).squeeze(-1)
        print("top 20 diff: ", (y / test_y).abs().flatten().topk(50).values)
        twox_diff = (y / test_y).abs() > 2
        print(">2x diff: ", twox_diff.sum().item())
        sixx_diff = (y / test_y).abs() > 6
        print(">6x diff: ", sixx_diff.sum().item())
        print("6x diff output vals: ", test_y[sixx_diff].flatten())

    # def inf_forward(self, x: Float[Tensor, "d_model"]) -> Float[Tensor, "d_model"]:
    #     sp = nn.Softplus()
    #     disc = sp(self.P + self.s_D(x).repeat(1, 1, self.d_model)) # size batch position d_model
    #     A_bar_pre = torch.unsqueeze(disc, -1) * self.A # size batch position d_model d_state
    #     A_bar = torch.exp(A_bar_pre)
    #     B = self.s_B(x) # size batch position d_state
    #     B_bar = (A_bar - 1) / A_bar_pre * (torch.unsqueeze(disc, -1) * torch.unsqueeze(B, 2)) # size batch position d_model d_state

    #     C = self.s_C(x)

    #     self.h = A_bar * self.h + B_bar * x
    #     return torch.matmul(self.h, C[:, None])


# %%
class Mamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        # dim: the dimension of the input
        # n_hidden: the dimension of the keys, queries, and values

        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        d_head = expand * d_state
        self.upscale = nn.Linear(d_model, d_head)
        self.gate = nn.Linear(d_model, d_head)
        self.conv = nn.Conv1d(d_head, d_head, d_conv, padding=d_conv-1, groups=d_head)
        self.ssm = SSM(d_head, d_state)
        self.downscale = nn.Linear(d_head, d_model)

        self.silu = nn.SiLU()

    def forward(self, x: Float[Tensor, "batch position d_model"], keep_hidden=False, use_hidden=False) -> Tuple[torch.Tensor, torch.Tensor]:
        upscaled = self.upscale(x)
        conv_out = self.conv(upscaled.transpose(1,2)).transpose(1,2)[:, :upscaled.shape[1]]
        ssm_out = self.ssm(self.silu(conv_out), keep_hidden=keep_hidden, use_hidden=use_hidden)
        gate_output = self.silu(self.gate(x))
        final_output = self.downscale(ssm_out * gate_output)

        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            raise ValueError("NaN or Inf values detected in Mamba output")

        return final_output
    
    def generate(self, x: Float[Tensor, "batch position d_model"], new_tokens: int):
        return self.forward(self.generate(x, new_tokens))

# %%
from einops import rearrange

class MambaLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_state, d_conv, expand):
        super().__init__()

        self.Heads = nn.ModuleList([Mamba(d_model, d_state, d_conv, expand) for _ in range(n_heads)])
        self.n_heads = n_heads
        self.rms_norm = nn.RMSNorm((d_model))
        self.out_project = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Float[Tensor, "batch position d_model"], keep_hidden=False, use_hidden=False) -> Float[Tensor, "batch position d_model"]:
        if torch.isinf(x).any():
            raise ValueError("Inf values detected in MambaLayer input")
        if torch.isnan(x).any():
            raise ValueError("NaN values detected in MambaLayer input")
        normed_x = self.rms_norm(x)
        head_outputs = torch.zeros(x.shape).to(x.device)
        for head in self.Heads:
            head_outputs += head(normed_x, keep_hidden=keep_hidden, use_hidden=use_hidden)
        x = x + self.layer_norm(self.out_project(head_outputs))
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN or Inf values detected in MambaLayer forward")
        return x
    
    def generate(self, x: Float[Tensor, "batch position d_model"], new_tokens: int):
        normed_x = self.rms_norm(x)
        
        out = self.forward(normed_x, keep_hidden=True)
        head_outputs = torch.zeros(x.shape).to(x.device)
        for head in self.Heads:
            head_outputs += head.generate(normed_x, new_tokens)
        x = x + self.layer_norm(self.out_project(head_outputs))
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN or Inf values detected in MambaLayer forward")
        return x

# %%
class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.net(x)

# %%
class MambaLM(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, d_model, d_state, d_conv, expand, context_len=1000):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.embedding.weight.requires_grad = False
        self.context_len = context_len

        self.pos_embedding = nn.Embedding(context_len, d_model)
        self.pos_embedding.weight.requires_grad = False
        self.pos_embedding.weight.data[:, ::2] = torch.sin(torch.arange(0, context_len)[:, None] / 10000 ** (torch.arange(0, d_model, 2)[None, :] / d_model))
        self.pos_embedding.weight.data[:, 1::2] = torch.cos(torch.arange(0, context_len)[:, None] / 10000 ** (torch.arange(1, d_model, 2)[None, :] / d_model))

        self.layers = nn.ModuleList([MambaLayer(n_heads, d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.n_heads = n_heads
        self.output_layer = nn.Linear(d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Count zeros on left side of each sequence
        mask = (x == 0).to(self.device())
        left_zeros = mask.cummin(dim=1)[0].sum(dim=1, keepdim=True)

        pos_embed_indices = torch.arange(x.shape[1]).expand(x.shape[0], -1).to(self.device()) - left_zeros
        pos_embed_indices = torch.where(pos_embed_indices >= 0, pos_embed_indices, 0)
        pos_embed = torch.where(pos_embed_indices.unsqueeze(-1) >= 0, self.pos_embedding(pos_embed_indices), 0)

        x = self.embedding(x)
        x = x + pos_embed
        if x.isnan().any() or x.isinf().any():
            raise ValueError("NaN or Inf values detected in MambaLM input")
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    @staticmethod
    def train_step(model, x: Int[Tensor, "batch position"], optimizer, scheduler, accelerator):
        optimizer.zero_grad()
        out = model.forward(x[:, :-1])
        ce = nn.CrossEntropyLoss()
        loss = ce(out.transpose(1,2), x[:, 1:])
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        # Check for nan gradients and zero them out
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                param.grad.zero_()
                # print(f"Gradient for {param.name} with {param.numel()} elements is {'nan' if torch.isnan(param.grad).any() else 'inf'}. Zeroing out.")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip the gradients
        optimizer.step()
        scheduler.step()
        return loss
    
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def train(model, ds, optimizer, scheduler, epochs=3, accelerator=None):
        best_loss = float('inf')
        is_main_process = accelerator is not None and accelerator.is_main_process
        progress_bar = tqdm(range(epochs), 
                           desc=f"Training (process {accelerator.process_index})", 
                           position=accelerator.process_index*2,
                           leave=True,
                           disable=not is_main_process)
        batch_progress = tqdm(total=len(ds), 
                             desc=f"Batches (process {accelerator.process_index})", 
                             position=accelerator.process_index*2+1,
                             leave=True,
                             disable=not is_main_process)
            
        for _ in progress_bar:
            epoch_loss = 0
            batch_progress.reset()

            for index, batch in enumerate(batch_progress):
                # if accelerator is not None and accelerator.is_main_process:
                loss = MambaLM.train_step(model, torch.stack(batch, dim=-1).to(next(model.parameters()).device), optimizer, scheduler, accelerator)
                epoch_loss += loss.item()
                batch_progress.update(1)
                    # print(f"Loss: {loss.item():0.4f}")
                # else:
                    # loss = MambaLM.train_step(model, torch.stack(batch, dim=-1).to(model.device()), optimizer, scheduler, accelerator)
                    # if index % 100 == 0:
                        # print(f"Loss: {loss.item():0.4f}")
                if index % 10 == 0 and is_main_process:
                    batch_progress.set_description(f"Loss: {epoch_loss / (index + 1):0.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if accelerator is not None and accelerator.is_main_process:
                    torch.save(model.state_dict(), "mamba_lm.pt")
        
        if is_main_process:
            print(f"Best loss: {best_loss:0.4f}")
                    
            # if accelerator is not None and accelerator.is_main_process:
            #     print(f"Epoch {_} complete")
            #     tokens = tokenizer.encode("hello there, happy world!")
            #     tokens = torch.tensor(tokens).unsqueeze(0).to(model.device())
            #     tokens = torch.cat([torch.zeros(1, 5, device=model.device(), dtype=tokens.dtype), tokens], dim=1)
            #     generation = model.forward(tokens)
            #     print(tokenizer.decode(generation[0].argmax(dim=-1).tolist()))

    def generate(self, x: Int[Tensor, "batch position"], new_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        returned = x
        out = self.forward(x, keep_hidden=True)
        next_token = out[:, -1].argmax(dim=-1)
        returned = torch.cat([returned, next_token], dim=1)

        for _ in range(new_tokens-1):
            out = self.forward(next_token, keep_hidden=True, use_hidden=True)
            next_token = out[:, -1].argmax(dim=-1)
            returned = torch.cat([returned, next_token], dim=1)
        return returned

    def generate_text(self, text, new_tokens):
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device())
        return tokenizer.decode(self.generate(tokens, new_tokens)[0])

# %% [markdown]
# # Get Data

# %%
import wget
import os
if not os.path.exists("input.txt"):
    wget.download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

with open('input.txt', 'r') as f:
    raw_text = f.read()
all_dialogues = raw_text.split('\n\n')

# %%
# type(tokenizer)

# %%
# type(all_dialogues)

# %% [markdown]
# ## Part 4.A

# %%
# tokenizer.encode("hello there, happy world!")

# %%
def num_params(model):
    return sum(p.numel() for p in list(model.parameters()))


# %%
torch.set_float32_matmul_precision('high')

# %%
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/shared/alex-zhao-storage/hf-cache")

# %%
# import numpy as np
# lengths = [len(text) for text in ds['train']['textbook']]
# print(f"Number of texts: {len(lengths)}")
# print(f"Mean length: {np.mean(lengths):.1f}")
# print(f"Std length: {np.std(lengths):.1f}") 
# print(f"Min length: {min(lengths)}")
# print(f"Max length: {max(lengths)}")
# print(f"Median length: {np.median(lengths):.1f}")

# %%
# L = torch.load('/shared/alex-zhao-storage/tiny-textbook-ds.pt')
# dataloader = DataLoader(L['input_ids'], batch_size=32, num_workers=8)

# %%
# model = MambaLM(tokenizer.vocab_size, 
#                 n_layers=12,
#                 n_heads=6,
#                 d_model=192,
#                 d_state=32,
#                 d_conv=4,
#                 expand=2,
#                 ).to('cuda')
# print(num_params(model))
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# model = torch.compile(model)
# MambaLM.train(model, dataloader, optimizer, scheduler, epochs=1)

# %%
# model = MambaLM(tokenizer.vocab_size, 
#                 n_layers=12,
#                 n_heads=6,
#                 d_model=192,
#                 d_state=32,
#                 d_conv=4,
#                 expand=2,
#                 ).to('cuda')
# print(num_params(model))
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# model = torch.compile(model)
# # L = tokenizer.batch_encode_plus(ds['train']['textbook'], padding=True, truncation=True, max_length=513, padding_side='left', return_tensors='pt').to('cuda')
# # torch.save(L, '/shared/alex-zhao-storage/tiny-textbook-ds.pt')
# # dataloader = DataLoader(L['input_ids'], batch_size=32)
# # L = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=65, padding_side='left').to('cuda')
# # dataloader = DataLoader(L['input_ids'], batch_size=4)
# model.my_train(dataloader, optimizer, scheduler)

# %%
def special_print(my_str, accelerator):
    if accelerator.is_main_process:
        print(my_str)
        print_time()

def print_time():
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S.%f")[:-3]
    print("Current Time =", current_time)

vocab_size = tokenizer.vocab_size

# L = tokenizer.batch_encode_plus(ds['train']['textbook'], padding=True, truncation=True, max_length=513, padding_side='left', return_tensors='pt').to('cuda')
# torch.save(L, '/shared/alex-zhao-storage/tiny-textbook-ds.pt')
# dataloader = DataLoader(L['input_ids'], batch_size=32)
# L = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=65, padding_side='left').to('cuda')
# dataloader = DataLoader(L['input_ids'], batch_size=4)

# model.my_train(dataloader, optimizer, scheduler)
def training_function(dataloader):
    accelerator = Accelerator()
    special_print("Accelerator initialized", accelerator)
    
    # 4. Prepare with Accelerator
    # L = torch.load('/shared/alex-zhao-storage/tiny-textbook-ds.pt')
    # special_print(f"Loaded dataset with {len(L['input_ids'])} samples", accelerator)
    # dataloader = DataLoader(L['input_ids'], batch_size=4, num_workers=0)
    # special_print(f"Loaded dataloader with {len(dataloader)} batches", accelerator)
    # L = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=65, padding_side='left').to('cuda')
    # dataloader = DataLoader(L['input_ids'], batch_size=4)

    model = MambaLM(vocab_size, 
                n_layers=16,
                n_heads=16,
                d_model=512,
                d_state=32,
                d_conv=8,
                expand=4,
                )
    
    special_print(f"Model params: {num_params(model)}", accelerator)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)
    special_print("Prepared with Accelerator", accelerator)
    model = torch.compile(model)
    special_print("Model compiled", accelerator)

    epochs = 1
    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(total=epochs * len(dataloader), desc="Training Progress")
        
    MambaLM.train(model.module, dataloader, optimizer, scheduler, epochs, accelerator)
    
    if accelerator.is_main_process:
        progress_bar.close()

    # 6. Save final model
    # Must unwrap to gather full weights from all shards
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "my_fsdp_fp32_model.pt")
    print("Model saved at my_fsdp_fp32_model.pt")

# %%
# from accelerate.logging import get_logger
# logger = get_logger(__name__)
# logger.setLevel("DEBUG")

# %%
# !export ACCELERATE_DEBUG_MODE=yes
# import torch.multiprocessing as mp

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    # mp.set_start_method("spawn", force=True)

    # Your main training code here
    # from accelerate import launch
    # launch()
    # model = MambaLM(vocab_size, 
    #             n_layers=12,
    #             n_heads=6,
    #             d_model=192,
    #             d_state=32,
    #             d_conv=4,
    #             expand=2,
    #             )
    # model = torch.compile(model)
    L = torch.load('/shared/alex-zhao-storage/tiny-textbook-ds.pt')
    print(f"Loaded dataset with {len(L['input_ids'])} samples")
    dataloader = DataLoader(L['input_ids'], batch_size=4, num_workers=0)
    print(f"Loaded dataloader with {len(dataloader)} batches")
    training_function(dataloader)

# %%
# %debug

# # %%
# ds = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/shared/alex-zhao-storage/hf-cache")
# L = tokenizer.batch_encode_plus(ds['train']['textbook'][:100], padding=True, truncation=True, max_length=513, padding_side='left')

# # %%
# L = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=33, padding_side='left')
# dataloader = DataLoader(L['input_ids'], batch_size=8)
# tokens = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=33, padding_side='left')
# notebook_launcher(training_function, args=(dataloader,), num_processes=8)

# # %%
# # import torch
# # import torch.nn as nn
# # from torch.profiler import profile, record_function, ProfilerActivity
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm
# # import torch.distributed
# # from accelerate import Accelerator, DistributedDataParallelKwargs

# # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

# # def training_function():
# #     # Initialize Accelerator (with DDP if used)
# #     accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
# #     model = MambaLM(
# #         tokenizer.vocab_size, 
# #         n_layers=12,
# #         n_heads=12,
# #         d_model=768,
# #         d_state=128,
# #         d_conv=4,
# #         expand=2,
# #     )

# #     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
# #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# #     # Tokenize and create DataLoader
# #     L = tokenizer.batch_encode_plus(all_dialogues, padding=True, truncation=True, max_length=50, padding_side='left')
# #     ds = DataLoader(L['input_ids'], batch_size=12)

# #     num_batches = len(ds)
# #     model, ds, optimizer, scheduler = accelerator.prepare(model, ds, optimizer, scheduler)
    
# #     epochs = 1
# #     ce = nn.CrossEntropyLoss()

# #     progress_bar = None
# #     if accelerator.is_main_process:
# #         progress_bar = tqdm(total=epochs * num_batches, desc="Training Progress")
    
# #     # ------------------------------
# #     # START PROFILER CONTEXT
# #     # ------------------------------
# #     with profile(
# #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
# #         record_shapes=True,      # (optional) get input shapes
# #         with_stack=True,         # (optional) gather stack traces
# #         profile_memory=True      # (optional) track tensor memory usage
# #     ) as prof:

# #         for epoch in range(epochs):
# #             for index, batch in enumerate(ds):
# #                 # Limit to 5 batches for a quick profiling run
# #                 if index >= 1:
# #                     break

# #                 x = torch.stack(batch, dim=-1)
                
# #                 optimizer.zero_grad()
# #                 with record_function("forward_pass"):
# #                     out = model(x[:, :-1])  # model forward

# #                 with record_function("loss_calc"):
# #                     loss = ce(out.transpose(1,2), x[:, 1:])
                
# #                 with record_function("backward"):
# #                     accelerator.backward(loss)
                
# #                 with record_function("optimizer_step"):
# #                     optimizer.step()
# #                     scheduler.step()

# #                 if accelerator.is_main_process:
# #                     progress_bar.update(1)

# #                 # Only call all_reduce if we're actually in a distributed context
# #                 if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
# #                     processed_batches = torch.tensor(1, device=accelerator.device)
# #                     torch.distributed.all_reduce(processed_batches, op=torch.distributed.ReduceOp.SUM)

# #                 print(f"Loss: {loss.item():0.4f}")
# #                 print(f"Batch {index} complete")
# #                 batch_sum = torch.sum(x.flatten()) % (10**7 + 9)
# #                 print(f"Batch sum mod 10^7+9: {batch_sum.item()}")

# #     # ------------------------------
# #     # END PROFILER CONTEXT
# #     # ------------------------------

# #     if accelerator.is_main_process:
# #         progress_bar.close()

# #     # Print a summary of the top CPU-consuming ops
# #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

# #     # Save final model
# #     unwrapped_model = accelerator.unwrap_model(model)
# #     torch.save(unwrapped_model.state_dict(), "my_fsdp_fp32_model.pt")
# #     print("Model saved at my_fsdp_fp32_model.pt")

# # %%
# from accelerate import notebook_launcher
# notebook_launcher(training_function, num_processes=8)

# # %%
# print("Model params", num_params(model))
# print("Layer params", num_params(model.layers[0]))
# print("Head params", num_params(model.layers[0].Heads[0]))
# print("SSM params", num_params(model.layers[0].Heads[0].ssm))
# print("Conv params", num_params(model.layers[0].Heads[0].conv))
# print("Up params", num_params(model.layers[0].Heads[0].upscale))
# print("Gate params", num_params(model.layers[0].Heads[0].gate))
# print("Down params", num_params(model.layers[0].Heads[0].downscale))
# print("RMS params", num_params(model.layers[0].rms_norm))
# print("Out params", num_params(model.layers[0].out_project))
# print("Layer norm params", num_params(model.layers[0].layer_norm))

# # %%
# torch.cuda.memory._record_memory_history(max_entries=100000)
# tokens = tokenizer.encode("hello there, happy world!")
# tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
# tokens = torch.cat([torch.zeros(1, 5, device='cuda', dtype=tokens.dtype), tokens], dim=1)
# generation = model.forward(tokens)
# print(tokenizer.decode(generation[0].argmax(dim=-1).tolist()))

# torch.cuda.memory._dump_snapshot("snapshot.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)


