import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm


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
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, skip_connection = False):
        super().__init__()
        self.skip_connection = skip_connection
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size= 3, padding = 1, bias = False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding = 1, bias = False),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x):
        if self.skip_connection:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim = 256):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, skip_connection=True),
            DoubleConv(in_channels, out_channels),
        )
        self.down_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )
    def forward(self, x, t):
        x = self.maxpool(x)
        emb = self.down_embed(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, skip_connection=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.up_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        emb = self.up_embed(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
class Unet(nn.Module):
    def __init__(self, input = 3 ,output = 3, time_dim = 256,device = 'cuda', num_class = None):
        super().__init__()
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.time_dim = time_dim
        self.inc = DoubleConv(input, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, output, kernel_size=1)

        if num_class is not None:
            self.label_emb = nn.Embedding(num_class, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / 10000**( torch.arange(0, channels, 2,  device=self.device).float() / channels)
        pos_en_even = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_en_odd = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_en = torch.cat([pos_en_even, pos_en_odd], dim = -1)
        return pos_en

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x5 = self.bot1(x4)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        return self.outc(x)

class Diffusion:
    def __init__(self, noise_steps = 500,  beta_start = 1e-4, beta_end = 0.02, image_size = 256, device = 'cuda'):
        super().__init__()
        self.image_size = image_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.noise_steps = noise_steps

        self.beta = self.schedule_noise_step().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim =0)

    def schedule_noise_step(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:None, None, None]
        e = torch.rand_like(x)
        return x * sqrt_alpha_hat + sqrt_one_minus_alpha_hat * e, e

    def sample_timestep(self, n):
        return torch.randint(1, self.noise_steps, (n,))

    def sample(self, model, n, labels, cfg_scale = 3): # sfg : classifier free guidance: make model more or less creative
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.image_size, self.image_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    unconditional_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    nosie = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * ( x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) /2
        x = (x * 255).type(torch.unit(8))
        return x





if __name__ == '__main__':
    net = Unet(num_class=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
























