import math
from functools import partial

import einops
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum
import opt_einsum as oe
import torchshow as ts
from torch import fft

from video_diffusion_pytorch.rotary_embedding_torch import RotaryEmbedding
from video_diffusion_pytorch.utils import default, rearrange_many, exists, prob_mask_like, convert_seg2onehot, \
    pseudo_image_from_semantic_map


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        # sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        sim = oe.contract('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, label_channel=4):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        # self.norm = SPADEGroupNorm(groups, dim_out, label_channel)

        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class SPADEGroupNorm(nn.Module):
    def __init__(self, groups, norm_nc, label_channel=4, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(groups, norm_nc)  # 32/16
        self.eps = eps
        nhidden = 32
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_channel, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, cond):
        x = self.norm(x)

        if cond is None:  # if not using segmap condition, SPADE group norm is the same as normal group norm
            return x

        cond = F.interpolate(cond, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return x * (1 + gamma) + beta


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, ):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class SemanticMapEmbedding(nn.Module):
    """
    Semantic map embedding module. it embeds a sequence of semantic maps into a vector in the latent space.
    """

    def __init__(self, dim, num_semantic_map_channels, hidden_dim=64,
                 attn_heads=4, attn_dim_head=32):
        super().__init__()
        self.num_semantic_map_channels = num_semantic_map_channels
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.init_conv = nn.Conv3d(num_semantic_map_channels, hidden_dim, 1)
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(hidden_dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))

        self.init_temporal_attn = Residual(PreNorm(hidden_dim, temporal_attn(hidden_dim)))

    def forward(self, x):
        if len(x.shape) == 4:  # if the input is indexed semantic map, convert it to one-hot
            x = torch.nn.functional.one_hot(x, num_classes=self.num_semantic_map_channels)
            x = rearrange(x, 'b f h w c -> b c f h w')
            x = x.float()
        x = self.init_conv(x)
        x = self.init_temporal_attn(x)
        return x


class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            num_semantic_map_channels=5,
            attn_heads=8,
            attn_dim_head=32,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=8
    ):
        super().__init__()
        self.channels = channels
        self.num_semantic_map_channels = num_semantic_map_channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads,
                                                      max_distance=32)  # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert init_kernel_size % 2 == 1, 'init kernel size must be odd'

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels + num_semantic_map_channels if num_semantic_map_channels else channels,
                                   init_dim, (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim)

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # semantic map embedding
        self.semantic_map_embedding = SemanticMapEmbedding(cond_dim,
                                                           num_semantic_map_channels=num_semantic_map_channels)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out,
                                                                 heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in,
                                                                heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
            self,
            x,
            time,
            cond_scale=2.,
            cond=None,
    ):
        logits = self.forward(x, time, cond, null_cond_prob=0.)
        return logits
        # if cond_scale == 1 or not self.has_cond:
        #     return logits
        
        # null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        # return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            x,
            time,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        if len(cond.shape) == 3 or cond.shape[1] == 1:
            cond = rearrange(cond, 'b h w -> b 1 h w') if len(cond.shape) == 3 else cond
            cond = repeat(cond, 'b 1 h w -> b t h w', t=x.shape[2])

        focus_present_mask = default(focus_present_mask,
                                     lambda: prob_mask_like((batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        cond = convert_seg2onehot(cond, num_classes=self.num_semantic_map_channels) if exists(cond) else None
        # concat cond with input
        x = torch.cat((x, cond), dim=1) if exists(cond) else x
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = self.semantic_map_embedding(cond)
            t = torch.cat((t, cond), dim=-1)

        h = []
        for block_idx, (block1, block2, spatial_attn, temporal_attn, downsample) in enumerate(self.downs):
            x = block1(x, time_emb=t, )
            x = block2(x, time_emb=t, )
            x = spatial_attn(x)
            # if block_idx == 1:
            #     x = self.eject_semantic_map(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, time_emb=t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, time_emb=t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
