import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

n_class = 10
batch_size = 16
patch_size = 16
projection_dim = 1024
mlp_hidden_dim = 4096
num_heads = 16
transformer_layers = 24

image_size = 32
num_patches = int((image_size*image_size)/(patch_size*patch_size)) + 1


class Gen_patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(3, projection_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class Cls_token(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, projection_dim))

    def forward(self, x):
        n_samples = x.shape[0]
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1) 

        return x

class PatchEncoder(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.position_embedding = nn.Embedding((num_patches), projection_dim)
        self.p_drop = nn.Dropout(dropout_rate)

    def forward(self, patch):
        positions = torch.arange(0, (num_patches)).to(device=DEVICE)
        encoded = patch + self.position_embedding(positions)
        encoded = self.p_drop(encoded)

        return encoded

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.attn = nn.GELU()
        self.fc2 = nn.Linear(out_channel, in_channel)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self, projection_dim, num_heads, mlp_hidden_dim, qkv_bias=False, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = SelfAttention(
                projection_dim, 
                heads=num_heads, 
                qkv_bias=qkv_bias, 
                dropout_rate=dropout_rate
        )

        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)

        self.mlp = MLP_block(
                in_channel=projection_dim,
                out_channel=mlp_hidden_dim,
                dropout_rate = dropout_rate
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class Mlp_head(nn.Module):
    def __init__(self, projection_dim, n_class, dropout_rate=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = nn.Linear(projection_dim, n_class)
        # self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        B, N, C = x.shape

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x


class Vision_transformer(nn.Module):
    def __init__(
            self,
            patch_size,
            n_class,
            projection_dim,
            depth,
            num_heads,
            qkv_bias=True,
            dropout_rate=0.0,
    ):
        super().__init__()
        self.patch_embed = Gen_patches(patch_size)
        self.position_patch_embed = PatchEncoder(dropout_rate=dropout_rate)
        self.cls_token = Cls_token(projection_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    projection_dim,
                    num_heads,
                    mlp_hidden_dim,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate
                )
                for _ in range(depth)
            ]
        )

        self.mlp_head = Mlp_head(projection_dim, n_class)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.position_patch_embed(x)
        

        for i, block in enumerate(self.blocks):
            x = block(x)

        output = self.mlp_head(x)

        return output

model = Vision_transformer(patch_size, n_class, projection_dim, transformer_layers, num_heads, dropout_rate=0.2)


