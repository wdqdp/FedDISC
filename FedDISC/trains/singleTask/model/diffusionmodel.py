import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.emb_layer_c = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t, c):
        x = self.maxpool_conv(x)
        emb_t = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        if c is not None:
            emb_c = self.emb_layer_c(c)[:, :, None].repeat(1, 1, x.shape[-1])
            return x + emb_t + emb_c
        else:
            return x + emb_t


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.emb_layer_c = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t, c):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb_t = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        if c is not None:
            emb_c = self.emb_layer_c(c)[:, :, None].repeat(1, 1, x.shape[-1])
            return x + emb_t + emb_c
        else:
            return x + emb_t
    

class UNet_conditional(nn.Module):
    def __init__(self, c_in=32, c_out=32, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # self.condition_time_proj = nn.Linear(c_in, time_dim)
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 24)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 12)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 6)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 12)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 24)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 48)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Linear(num_classes, time_dim)
        
        # 新增：交叉注意力模块，用于融合条件信息
        # 注意：这里 embed_dim 选择为 c_in，因为我们希望对输入最初的通道数进行注意力融合
        self.cross_attn = nn.MultiheadAttention(embed_dim=c_in, num_heads=4, batch_first=False)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, condition):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # (bs, 256)

        if condition is not None:
            # t += self.label_emb(y)
            # x += condition

            # cond_avg = condition.mean(dim=-1)
            # # 投影到 time_dim
            # cond_proj = self.condition_time_proj(cond_avg)
            # # 将条件信息融合到时间编码
            # t = t + cond_proj

            # 将为标签融合到时间编码
            cond_emb = self.label_emb(condition)  # (bs, time_dim)
            t = t + cond_emb

            # query = x.permute(2, 0, 1)      # (length, bs, c_in)
            # key = condition.permute(2, 0, 1)  # (length, bs, c_in)
            # value = condition.permute(2, 0, 1)  # (length, bs, c_in)
            # # 得到交叉注意力输出，注意这里不需要 mask
            # attn_out, _ = self.cross_attn(query, key, value)
            # # 将输出转换回 (bs, c_in, length)
            # attn_out = attn_out.permute(1, 2, 0)
            # # 利用残差连接将注意力结果融合到原始输入中
            # x = x + attn_out
        else:
            cond_emb = condition

        x1 = self.inc(x)     # (bs, 64, 48)
        x2 = self.down1(x1, t, cond_emb) # (bs, 128, 24)
        x2 = self.sa1(x2)     # (bs, 128, 24)
        x3 = self.down2(x2, t, cond_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, cond_emb)
        x4 = self.sa3(x4)   # (3, 256, 6)

        x4 = self.bot1(x4)  # (3, 512, 6)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)  # (3, 256, 6)

        x = self.up1(x4, x3, t, cond_emb)  # x4:(3, 256, 6) x3:(3, 256, 12) t:(3,256)  x:(bs,128,12)
        x = self.sa4(x)
        x = self.up2(x, x2, t, cond_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t, cond_emb)
        x = self.sa6(x)   # (bs,64,48)
        output = self.outc(x) # (3, 32, 48)
        return output


# class UNet_conditional(nn.Module):
#     def __init__(self, c_in=32, c_out=32, time_dim=256, num_classes=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
        
#         # 下采样部分（编码器）
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 24)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 12)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 6)

#         # Bottleneck部分
#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)

#         # 上采样部分（解码器）
#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 12)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 24)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 48)
#         self.outc = nn.Conv1d(64, c_out, kernel_size=1)

#         # 如果提供了条件（例如类别信息），构造条件相关的嵌入层
#         # 这里分为两部分：
#         # 1. 将条件信息融入到时间编码中（高层语义指导生成过程）
#         # 2. 在上采样阶段注入额外的上下文信息，调制解码器特征
#         if num_classes is not None:
#             self.time_emb = nn.Linear(num_classes, time_dim)
#             # self.context_emb1 = nn.Linear(num_classes, 128)
#             # self.context_emb2 = nn.Linear(num_classes, 64)
#             # self.context_emb3 = nn.Linear(num_classes, 64)
        
#         # 可选：交叉注意力模块（这里暂时不主动使用）
#         self.cross_attn = nn.MultiheadAttention(embed_dim=c_in, num_heads=4, batch_first=False)

#     def pos_encoding(self, t, channels):
#         # 位置编码（类似Transformer中的正余弦编码）
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t, condition):
#         # 对时间t进行位置编码，注意这里不再与条件融合
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)  # (bs, time_dim)

#         if condition is not None:
#             # 将为标签融合到时间编码
#             cond_emb = self.label_emb(condition)  # (bs, time_dim)
#             t = t + cond_emb
        
#         # 下采样路径（编码器）：不注入条件信息
#         x1 = self.inc(x)            # (bs, 64, seq_len)
#         x2 = self.down1(x1, t)        # (bs, 128, seq_len//2)
#         x2 = self.sa1(x2)           # (bs, 128, seq_len//2)
#         x3 = self.down2(x2, t)        # (bs, 256, seq_len//4)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)        # (bs, 256, seq_len//8)
#         x4 = self.sa3(x4)

#         # Bottleneck
#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)          # (bs, 256, seq_len//8)

#         # 上采样路径（解码器）：在每个阶段用条件信息乘以特征进行融合
#         x = self.up1(x4, x3, t)      # (bs, 128, seq_len//4)
#         if condition is not None:
#             context1 = self.context_emb1(condition).unsqueeze(-1)  # (bs, 128, 1)
#             x = x * context1
#         x = self.sa4(x)
        
#         x = self.up2(x, x2, t)       # (bs, 64, seq_len//2)
#         if condition is not None:
#             context2 = self.context_emb2(condition).unsqueeze(-1)  # (bs, 64, 1)
#             x = x * context2
#         x = self.sa5(x)
        
#         x = self.up3(x, x1, t)       # (bs, 64, seq_len)
#         if condition is not None:
#             context3 = self.context_emb3(condition).unsqueeze(-1)  # (bs, 64, 1)
#             x = x * context3
#         x = self.sa6(x)

#         output = self.outc(x)        # (bs, c_out, seq_len)
#         return output
    


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, dim=32, seq=48, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.dim = dim
        self.seq = seq
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.dim, self.seq)).to(self.device)
            # for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    

def loss_diff(model, x, condition):
    mse = nn.MSELoss()
    diffusion = Diffusion()
    t = diffusion.sample_timesteps(x.shape[0]).to(device)
    x_t, noise = diffusion.noise_images(x, t)
    predicted_noise = model(x_t, t, condition)
    loss = mse(noise, predicted_noise)

    return loss






if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=3, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    # x = torch.randn(3, 3, 64, 64)
    x = torch.randn(3, 32, 48)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    condition = torch.randn(3, 3)
    print(net(x, t, condition).shape)