import os
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# 1D卷积标准化层
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# 注意力块：修改为支持序列数据
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape  # 这里L是序列长度
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, L).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, L)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)
    

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t, y):
        """
        Apply the module to `x` given `t` timestep embeddings, `y` conditional embedding same shape as t.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x, t, y):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t, y)
            else:
                x = layer(x)
        return x
    

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
    
class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val
    

class LayerNorm(nn.Module):
    def __init__(self, feats, stable=True, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim
        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
    

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        norm=True,
    ):
        super().__init__()
        # 修改为1D的GroupNorm
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()  # 保留SiLU激活函数
        # 修改为1D的卷积层
        self.project = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        # 进行归一化
        x = self.groupnorm(x)

        # 如果scale_shift提供，则进行缩放和偏移
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # 激活函数
        x = self.activation(x)

        # 通过1D卷积进行投影
        return self.project(x)
    

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=4,
        heads=8,
        norm_context=False,
        cosine_sim_attn=False
    ):
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        # 如果没有提供 context_dim，则默认为 dim
        context_dim = dim if context_dim is None else context_dim

        # 1D的LayerNorm
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        # 定义参数
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))  # 用于 classifier free guidance
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # 查询线性变换
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)  # 键值线性变换

        # 输出的线性变换
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)  # 1D归一化
        )

    def forward(self, x, context):
        # 输入形状：batch_size, dim, seq_len
        b, n, device = *x.shape[:2], x.device

        # 对x和context进行LayerNorm
        x = self.norm(x)
        context = self.norm_context(context)

        # 将查询q、键k和值v计算出来
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        # 将q, k, v的形状调整为适应多头注意力：b, h, n, d
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # 添加null key / value，用于classifier free guidance
        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # 缩放查询q
        q = q * self.scale

        # 计算q与k之间的相似度
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale

        # 进行softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        # 根据注意力得分加权v
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # 恢复输出的形状：b, n, h * d
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 通过to_out进行输出投影
        return self.to_out(out)


class GlobalContext(nn.Module):
    """ A superior form of squeeze-excitation with an attention mechanism for 1D data """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()

        # 1D卷积：将输入通道数映射到一个通道为1的上下文信息
        self.to_k = nn.Conv1d(dim_in, 1, 1)

        # 隐藏层通道数：设为dim_out的一半或至少为3
        hidden_dim = max(3, dim_out // 2)

        # 定义网络
        self.net = nn.Sequential(
            # 第一层1D卷积，将输入的dim_in映射到hidden_dim
            nn.Conv1d(dim_in, hidden_dim, 1),
            nn.SiLU(),  # 激活函数
            # 第二层1D卷积，将hidden_dim映射到dim_out
            nn.Conv1d(hidden_dim, dim_out, 1),
            nn.Sigmoid()  # Sigmoid输出，得到注意力系数
        )

    def forward(self, x):
        # 提取上下文信息
        context = self.to_k(x)  # [batch_size, 1, seq_len]

        # 重新排列x和context，确保其形状适用于计算
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')

        # 加权计算
        out = torch.einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)

        # 重新排列输出结果，适配维度
        out = rearrange(out, '... -> ... 1')

        # 通过网络处理结果
        return self.net(out)
    

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, use_global_context=False, groups=8):
        super().__init__()

        self.block1 = Block(in_channels, out_channels, groups=groups)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels * 2)  # 输出维度是2倍的out_channels
        )

        self.block2 = Block(out_channels, out_channels, groups=groups)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        cond_dim = time_channels
        self.gca = GlobalContext(dim_in=out_channels, dim_out=out_channels) if use_global_context else Always(1)

        # 修复CrossAttention，确保处理的是正确形状
        self.cross_attn = CrossAttention(dim=out_channels, context_dim=cond_dim, cosine_sim_attn=False)

    def forward(self, x, t, y):
        """
        `x` has shape `[batch_size, dim, seq_len]`
        `t` has shape `[batch_size, time_dim]`
        `y` has shape `[batch_size, num_time_tokens, cond_dim]`
        """
        h = self.block1(x)

        # 通过cross attention应用条件y
        context = y
        size = h.size(-1)  # seq_len
        hidden = rearrange(h, 'b c l -> b l c')  # 改为 [batch_size, seq_len, channels] 形状
        attn = self.cross_attn(hidden, context)
        attn = rearrange(attn, 'b l c -> b c l', l=size)  # 重新调整回 [batch_size, channels, seq_len]
        h += attn  # residual connection

        # 添加时间步嵌入
        t = self.time_emb(t)
        t = rearrange(t, 'b c -> b c 1')  # Time step embedding reshaped for broadcasting
        scale_shift = t.chunk(2, dim=1)  # Split the time embedding into scale and shift
        h = self.block2(h, scale_shift=scale_shift)

        # 应用global context
        h *= self.gca(h)
        return h + self.shortcut(x)

    

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=32,           # 设置为32, 根据输入形状
        model_channels=128,
        out_channels=32,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        label_num=4,
        num_time_tokens=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 2
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # condition embedding
        cond_dim = time_embed_dim
        self.label_embedding = nn.Linear(label_num, time_embed_dim)     # LABEL (bs,label_num)
        # self.label_embedding = nn.Embedding(label_num, time_embed_dim)    # LABEL (bs,)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_embed_dim, num_time_tokens * cond_dim),
            Rearrange('b (r d) -> b r d', r=num_time_tokens)
        )

        # down blocks
        self.down_blocks = nn.ModuleList([ 
            TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))  # Conv1d instead of Conv2d
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))  # 1D Downsample
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))  # 1D Upsample
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1),  # Conv1d instead of Conv2d
        )

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor):
        """Apply the model to an input batch.

        Args:
            x (Tensor): [N x C x L] where L is the sequence length
            t (Tensor): [N,] a 1-D batch of timesteps.
            y (Tensor): [N,] LongTensor conditional labels.

        Returns:
            Tensor: [N x C x ...]
        """
        # time step embedding
        t = self.time_embed(timestep_embedding(t, self.model_channels))
        y = self.label_embedding(y)
        y = self.to_time_tokens(y)

        hs = []
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, t, y)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t, y)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t, y)
        return self.out(h)
    

# ======diffusion model======
def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t, y)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, y, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample(self, model: nn.Module, y: torch.LongTensor, seq_len, batch_size=8, channels=32):
        # sample new images
        # denoise: reverse diffusion
        shape = (batch_size, channels, seq_len)
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)  # x_T ~ N(0, 1)
        imgs = []
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y)
            # imgs.append(img.cpu().numpy())
        return img

    def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor):
        # compute train losses
        noise = torch.randn_like(x_start)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(x_start, t, noise=noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t, y)  # predict noise from noisy image and condition
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        y: torch.LongTensor,
        seq_len,
        batch_size=8,
        channels=32,
        ddim_timesteps=50,
        ddim_discr_method="quad",
        ddim_eta=0.0,
        clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, channels, seq_len), device=device)
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_noise = model(sample_img, t, y)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            
        return sample_img










if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNetModel().to(device)
    x = torch.randn(3, 32, 48).to(device)
    condition = torch.randn(3, 4).to(device)
    timesteps = 1000
    batch_size = 3
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    # loss = gaussian_diffusion.train_losses(net, x, t, condition)
    # print(loss.item())
    generated_images = gaussian_diffusion.ddim_sample(net, condition, 48, batch_size=batch_size, channels=32)
    print(generated_images.shape)



