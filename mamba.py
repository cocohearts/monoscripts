# %%
import wandb
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, Optional, List
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from typing import Callable
from jaxtyping import Float, Int
import einops
from dataclasses import dataclass
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs, notebook_launcher
import ipywidgets as widgets
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer

# %%
import dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
import os
dotenv.load_dotenv()
# import huggingface_hub
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# huggingface_hub.login(token=HUGGINGFACE_API_KEY)


# ds = load_dataset("bigcode/the-stack-v2", cache_dir="/shared/alex-zhao-storage/the-stack-v2", split="train")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11b-Vision", cache_dir="/shared/alex-zhao-storage/hf-cache")
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="/shared/alex-zhao-storage/hf-cache")
tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/shared/alex-zhao-storage/hf-cache")
# tokenizer.vocab_size
# torch._logging.set_log_level("ERROR")

# %%
def is_in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            return True
        return False
    except ImportError:
        return False

if is_in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# %%
def num_params(model):
    return sum(p.numel() for p in list(model.parameters()))

# %%
debug = False
if debug:
    print(tokenizer.decode(tokenizer.encode("hello there, happy world! test lol lol")))

# %%
# ds = load_dataset("HuggingFaceFW/fineweb-edu", "default")
# L = torch.load('/shared/alex-zhao-storage/tiny-textbook-ds.pt')
# dataloader = DataLoader(L['input_ids'], batch_size=32, num_workers=8)
class TextDataset(Dataset):
    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        output = self.tokenizer.encode(self.text[idx], return_tensors="pt", padding=True, truncation=True, padding_side="left", max_length=512)
        # Squeeze to remove batch dimension added by tokenizer
        output = output.squeeze(0)

        # Pad to length 513 from the left if needed
        padding_length = 512 - output.size(0)
        padding = torch.full((padding_length,), self.tokenizer.pad_token_id)
        output = torch.cat([padding, output], dim=0)
            
        return output
    
def get_dataloader(batch_size=16, accelerator=None):
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/shared/alex-zhao-storage/hf-cache")
    # dataset = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/shared/alex-zhao-storage/hf-cache")
    is_main_process = accelerator is None or accelerator.is_main_process
    if not is_main_process:
        dataset = load_dataset("DKYoon/SlimPajama-6B", cache_dir="/shared/alex-zhao-storage/hf-cache", split="train", download_config=DownloadConfig(disable_tqdm=True))
    else:
        dataset = load_dataset("DKYoon/SlimPajama-6B", cache_dir="/shared/alex-zhao-storage/hf-cache", split="train")
    return DataLoader(TextDataset(dataset['text'], tokenizer), batch_size=batch_size, num_workers=0)

# %%
if debug:
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch.shape)
        break

# %%
tokenizer.vocab_size

# %% [markdown]
# # Prefix Ops

# %%
from math import log2, ceil
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
        n_layers = ceil(log2(t.shape[1]))
        len_diff = 2**n_layers - t.shape[1]
        t = torch.cat([t, torch.zeros(t.shape[0], len_diff, t.shape[2], t.shape[3]).to(t.device)], dim=1)
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
        output = down_tensors[-1] + t
        return output[:, :t.shape[1] - len_diff]

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

        if pref_sum and not use_hidden:
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
                h = AinvsBsum * torch.exp(A_bar_prod_log)
                y = torch.matmul(h, C[..., None]).squeeze(-1)
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
            self.h = h[:, -1:]
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
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.context_len = context_len

        self.embedding = nn.Embedding(self.vocab_size, d_model)

        self.pos_embedding = nn.Embedding(self.context_len, d_model)

        self.pos_embedding.weight.data[:, ::2] = torch.sin(torch.arange(0, self.context_len)[:, None] / 10000 ** (torch.arange(0, self.d_model, 2)[None, :] / self.d_model))
        self.pos_embedding.weight.data[:, 1::2] = torch.cos(torch.arange(0, self.context_len)[:, None] / 10000 ** (torch.arange(1, self.d_model, 2)[None, :] / self.d_model))

        self.layers = nn.ModuleList([MambaLayer(n_heads, d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.output_layer = nn.Linear(d_model, self.vocab_size)

    def forward(self, x: torch.Tensor, keep_hidden=False, use_hidden=False, position_shift=0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Count zeros on left side of each sequence
        mask = (x == 0).to(next(self.parameters()).device)
        left_zeros = mask.cummin(dim=1)[0].sum(dim=1, keepdim=True)

        pos_embed_indices = torch.arange(x.shape[1]).expand(x.shape[0], -1).to(next(self.parameters()).device) - left_zeros
        pos_embed_indices = torch.where(pos_embed_indices >= 0, pos_embed_indices, 0)
        pos_embed_indices += position_shift
        pos_embed = torch.where(pos_embed_indices.unsqueeze(-1) >= 0, self.pos_embedding(pos_embed_indices), 0)

        x = self.embedding(x)
        x = x + pos_embed
        if x.isnan().any() or x.isinf().any():
            raise ValueError("NaN or Inf values detected in MambaLM input")
        for layer in self.layers:
            x = layer(x, keep_hidden=keep_hidden, use_hidden=use_hidden)
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
    
    @staticmethod
    def train(model, ds, optimizer, scheduler, epochs=1, accelerator=None, enable_wandb=False):
        best_loss = float('inf')
        is_main_process = accelerator is not None and accelerator.is_main_process
        progress_bar = tqdm(range(epochs), 
                           desc=f"Training (process {accelerator.process_index})", 
                           position=accelerator.process_index*2,
                           leave=True,
                           disable=not is_main_process)
        batch_progress = tqdm(ds, 
                             desc=f"Batches (process {accelerator.process_index})", 
                             position=accelerator.process_index*2+1,
                             leave=True,
                             disable=not is_main_process)
            
        for _ in progress_bar:
            epoch_loss = 0
            batch_progress.reset()

            for index, batch in enumerate(batch_progress):
                # if accelerator is not None and accelerator.is_main_process:
                loss = MambaLM.train_step(model, batch.to(next(model.parameters()).device), optimizer, scheduler, accelerator)
                epoch_loss += loss.item()
                batch_progress.update(1)
                batch_progress.set_description(f"Loss: {epoch_loss / (index + 1):0.4f}")
                if enable_wandb:
                    wandb.log({"loss": loss.item()})

                # if index % 100 == 0 and is_main_process:
                #     print(f"Test generation `the weather today is` at {index / len(ds) * 100:.2f}% completion: ", MambaLM.generate_text(model, "the weather today is", 10))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if accelerator is not None and accelerator.is_main_process:
                    torch.save(model.state_dict(), "mamba_lm.pt")

        if is_main_process:
            print(f"Best loss: {best_loss:0.4f}")
                    
            
    def generate(self, x: Int[Tensor, "batch position"], new_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        returned = x

        out = self.forward(x, keep_hidden=True)
        next_token = out[:, -1].argmax(dim=-1)[None, :]
        returned = torch.cat([returned, next_token], dim=1)

        for _ in range(new_tokens-1):
            out = self.forward(next_token, keep_hidden=True, use_hidden=True)
            next_token = out[:, -1].argmax(dim=-1)[None, :]
            returned = torch.cat([returned, next_token], dim=1)
        return returned

    def generate_text(self, text, new_tokens):
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(next(self.parameters()).device)
        return tokenizer.decode(MambaLM.generate(self, tokens, new_tokens)[0])

# %% [markdown]
# # Get Data

# %%
# import wget
# import os
# if not os.path.exists("input.txt"):
#     wget.download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# with open('input.txt', 'r') as f:
#     raw_text = f.read()
# all_dialogues = raw_text.split('\n\n')

# %%
if debug:
    ds = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/shared/alex-zhao-storage/hf-cache")
    print(tokenizer.encode("hello there, happy world!"))
    lengths = [len(text) for text in ds['train']['textbook']]
    print(f"Number of texts: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Std length: {np.std(lengths):.1f}") 
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Median length: {np.median(lengths):.1f}")

# %%
# L = torch.load('/shared/alex-zhao-storage/tiny-textbook-ds.pt')
# special_print(f"Loaded dataset with {len(L['input_ids'])} samples", accelerator)
# dataloader = DataLoader(L['input_ids'], batch_size=1, num_workers=0)
# special_print(f"Loaded dataloader with {len(dataloader)} batches", accelerator)

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

def training_function():
    accelerator = Accelerator()
    special_print("Accelerator initialized", accelerator)
    is_main_process = accelerator.is_main_process
    if is_main_process:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="monoscripts-mamba")

    dataloader = get_dataloader(4, accelerator)
    special_print(f"Loaded dataloader with {len(dataloader)} batches", accelerator)
    for batch in dataloader:
        tokens_per_batch = batch.numel()
        break
    num_tokens = len(dataloader) * tokens_per_batch
    special_print(f"Number of tokens: {num_tokens}", accelerator)
    
    # model = MambaLM(vocab_size, 
    #             n_layers=12,
    #             n_heads=6,
    #             d_model=192,
    #             d_state=32,
    #             d_conv=4,
    #             expand=2,
    #             )

    model = MambaLM(vocab_size, 
                n_layers=12,
                n_heads=12,
                d_model=768,
                d_state=64,
                d_conv=4,
                expand=2,
                )
    
    special_print(f"Model params: {num_params(model)}", accelerator)
    special_print(f"Token:params ratio: {num_tokens / num_params(model)}", accelerator)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)
    special_print("Prepared with Accelerator", accelerator)
    model = torch.compile(model)
    special_print("Model compiled", accelerator)
    
    if is_main_process:
        wandb.watch(model)

    epochs = 1
    MambaLM.train(model, dataloader, optimizer, scheduler, epochs, accelerator, enable_wandb=is_main_process)
    
    # 6. Save final model
    # Must unwrap to gather full weights from all shards
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "my_fsdp_fp32_model.pt")
    print("Model saved at my_fsdp_fp32_model.pt")

# %%
if debug:
    model = MambaLM(vocab_size, 
                    n_layers=12,
                    n_heads=12,
                    d_model=768,
                    d_state=64,
                    d_conv=4,
                    expand=2,
                ).to('cuda')
    print(num_params(model))
    model.generate_text("hello there, happy world! test lol lol", 10)

# %%
torch.set_float32_matmul_precision('high')

# %%
# training_function()

# %%
nb_parallelize = False

if __name__ == '__main__':
    # Check if we're running in a notebook or regular Python script
    if is_in_notebook():
        if nb_parallelize:
            notebook_launcher(training_function, num_processes=8)
        else:
            notebook_launcher(training_function, num_processes=1)
    else:
        training_function()

# %%
print("Model params", num_params(model))
print("Layer params", num_params(model.layers[0]))
print("Head params", num_params(model.layers[0].Heads[0]))
print("SSM params", num_params(model.layers[0].Heads[0].ssm))
print("Conv params", num_params(model.layers[0].Heads[0].conv))
print("Up params", num_params(model.layers[0].Heads[0].upscale))
print("Gate params", num_params(model.layers[0].Heads[0].gate))
print("Down params", num_params(model.layers[0].Heads[0].downscale))
print("RMS params", num_params(model.layers[0].rms_norm))
print("Out params", num_params(model.layers[0].out_project))
print("Layer norm params", num_params(model.layers[0].layer_norm))


