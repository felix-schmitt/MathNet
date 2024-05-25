from torch import nn
from model_parts.CNNEncoder import positionalencoding2d
from vit_pytorch.cvt import group_by_key_prefix_and_remove_prefix, LayerNorm, Transformer

class CvT(nn.Module):
    def __init__(
        self,
        *,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 1
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        latents = self.layers(x)
        positional_encoding = positionalencoding2d(latents.size(1), latents.size(2), latents.size(3))
        positional_encoding = positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.to(latents.device)
        positional_encoding = latents + positional_encoding
        positional_encoding = positional_encoding.view(positional_encoding.size(0), positional_encoding.size(1), -1)
        positional_encoding = positional_encoding.permute(2, 0, 1)
        return positional_encoding
