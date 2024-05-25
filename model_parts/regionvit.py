from torch import nn
from einops.layers.torch import Rearrange, Reduce
from vit_pytorch.regionvit import cast_tuple, Downsample, PEG, R2LTransformer, divisible_by
class RegionViT(nn.Module):
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        window_size = 7,
        num_classes = 1000,
        tokenize_local_3_conv = False,
        local_patch_size = 4,
        use_peg = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3,
    ):
        super().__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'

        self.local_patch_size = local_patch_size

        region_patch_size = local_patch_size * window_size
        self.region_patch_size = local_patch_size * window_size

        init_dim, *_, last_dim = dim

        # local and region encoders

        if tokenize_local_3_conv:
            self.local_encoder = nn.Sequential(
                nn.Conv2d(channels, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            )
        else:
            self.local_encoder = nn.Conv2d(channels, init_dim, 8, 4, 3)

        self.region_encoder = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = region_patch_size, p2 = region_patch_size),
            nn.Conv2d((region_patch_size ** 2) * channels, init_dim, 1)
        )

        # layers

        current_dim = init_dim
        self.layers = nn.ModuleList([])

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and use_peg

            self.layers.append(nn.ModuleList([
                Downsample(current_dim, dim) if need_downsample else nn.Identity(),
                PEG(dim) if need_peg else nn.Identity(),
                R2LTransformer(dim, depth = num_layers, window_size = window_size, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

            current_dim = dim

        # final logits

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        *_, h, w = x.shape
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'

        local_tokens = self.local_encoder(x)
        region_tokens = self.region_encoder(x)

        for down, peg, transformer in self.layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)

        return self.to_logits(region_tokens)