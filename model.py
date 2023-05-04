import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from tqdm import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device)
                        * -(math.log(10000) / half_dim))
        emb = torch.outer(x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        assert h % 2 == 0 or w % 2 == 0, "w and h must be even"
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_dim, num_groups=32, activatedfun=nn.SiLU):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            activatedfun(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            activatedfun(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.time_layer = nn.Sequential(
            activatedfun(),
            nn.Linear(time_dim, out_channels),
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        output = self.layer1(x)
        output += self.time_layer(time_emb)[:, :, None, None]
        output = self.layer2(output) + self.shortcut(x)

        return output


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, channel_mults=(1, 2, 4), num_res_blocks=2, time_dim=128 * 4, activatedfun=nn.SiLU, dropout=0.1, num_groups=32):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResidualBlock(now_channels, out_channels, dropout, time_dim=time_dim,
                                  activatedfun=activatedfun, num_groups=num_groups)
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downblocks.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim,
                          activatedfun=activatedfun, num_groups=num_groups),
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim,
                          activatedfun=activatedfun, num_groups=num_groups),
        ])

        self.upblocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResidualBlock(channels.pop() + now_channels, out_channels, dropout,
                                  time_dim=time_dim, activatedfun=activatedfun, num_groups=num_groups)
                )
                now_channels = out_channels

            if i != 0:
                self.upblocks.append(Upsample(now_channels))

        assert len(channels) == 0

        self.last_layer = nn.Sequential(
            nn.GroupNorm(num_groups, base_channels),
            activatedfun(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

    def forward(self, x, time):
        time_emb = self.time_embedding(time)
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downblocks:
            x = layer(x, time_emb)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)

        x = self.last_layer(x)
        assert len(skips) == 0
        return x


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, img_channels, betas):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.img_channels = img_channels
        self.num_timesteps = len(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas",
                             to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(
            betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, t):
        return ((x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t)) * extract(self.reciprocal_sqrt_alphas, t, x.shape))

    @torch.no_grad()
    def sample(self, batch_size, device):
        x = torch.randn(batch_size, self.img_channels,
                        *self.image_size, device=device)

        for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc='sampling loop time step', total=self.num_timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * \
                    torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_grid(self, batch_size, process_step, device):
        x = torch.randn(batch_size, self.img_channels,
                        *self.image_size, device=device)
        process_x = [x]
        interval = self.num_timesteps / (process_step - 1)
        i = 1
        for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc='sampling loop time step', total=self.num_timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * \
                    torch.randn_like(x)

            if i >= interval * len(process_x) or t == 0:
                process_x.append(x)
            i += 1

        process_x = torch.stack(process_x, dim=0).view(-1, 3, *self.image_size)
        return process_x.cpu().detach()

    def perturb_x(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)

    def compute_loss(self, x, t):
        noise = torch.randn_like(x)
        pred = self.perturb_x(x, t, noise)
        pred_noise = self.model(pred, t)
        return F.mse_loss(pred_noise, noise)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.compute_loss(x, t)


class DDIM(GaussianDiffusion):
    @torch.no_grad()
    def sample(self, batch_size, ddim_timesteps=20, ddim_eta=0.0, device="cuda"):
        ddim_timestep_seq = np.asarray(
            list(range(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(
            np.array([0]), ddim_timestep_seq[:-1])
        x = torch.randn(batch_size, self.img_channels,
                                 *self.image_size, device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t_batch = torch.tensor(
                [ddim_timestep_seq[i]], device=device, dtype=torch.long).repeat(batch_size)
            prev_t_batch = torch.tensor(
                [ddim_timestep_prev_seq[i]], device=device, dtype=torch.long).repeat(batch_size)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(
                self.alphas_cumprod, t_batch, x.shape)
            alpha_cumprod_t_prev = extract(
                self.alphas_cumprod, prev_t_batch, x.shape)

            # 2. predict noise using model
            pred_noise = self.model(x, t_batch)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t))
                    * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # 4. compute variance: "sigma_t(η)"
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t"
            pred_dir_xt = torch.sqrt(
                1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

            # 6. compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev

        return x.cpu().detach()
    
    @torch.no_grad()
    def sample_grid(self, batch_size, process_step, device, ddim_timesteps, ddim_eta=0.0):
        x = torch.randn(batch_size, self.img_channels,
                        *self.image_size, device=device)
        process_x = [x]
        interval = ddim_timesteps / (process_step - 1)
        
        #------------same sample function ---------------------------------------
        ddim_timestep_seq = np.asarray(
            list(range(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(
            np.array([0]), ddim_timestep_seq[:-1])
        x = torch.randn(batch_size, self.img_channels,
                                 *self.image_size, device=device)
        step = 1

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t_batch = torch.tensor(
                [ddim_timestep_seq[i]], device=device, dtype=torch.long).repeat(batch_size)
            prev_t_batch = torch.tensor(
                [ddim_timestep_prev_seq[i]], device=device, dtype=torch.long).repeat(batch_size)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(
                self.alphas_cumprod, t_batch, x.shape)
            alpha_cumprod_t_prev = extract(
                self.alphas_cumprod, prev_t_batch, x.shape)

            # 2. predict noise using model
            pred_noise = self.model(x, t_batch)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t))
                    * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # 4. compute variance: "sigma_t(η)" 
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" 
            pred_dir_xt = torch.sqrt(
                1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

            # 6. compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                pred_dir_xt + sigmas_t * torch.randn_like(x)
            
            x = x_prev
            
            if step >= interval * len(process_x) or i == 0:
                process_x.append(x)
            step += 1

        process_x = torch.stack(process_x, dim=0).view(-1, 3, *self.image_size)
        return process_x.cpu().detach()


class EMA():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, old, new):
        old_dict = old.state_dict()
        new_dict = new.state_dict()
        for key in old_dict.keys():
            new_dict[key].data = old_dict[key].data * \
                self.decay + new_dict[key].data * (1 - self.decay)
