from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).typed_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024), #推入的序列长
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # base^{2i/d}

    if rope_scaling is not None:
        original_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

        if end / original_max > 1.0:
            corr_dim = next(
                (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max),
                dim // 2,
            )

            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
                dim // 2 - 1, 1
            )

            beta = beta_slow + (beta_fast - beta_slow) * power

            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )

            freqs = freqs * scale

        t = torch.arange(end, device=freqs.device) #生成位置索引序列 m

        # 计算每个位置的角度：position * frequency
        # 外积得到 (seq_len, dim // 2) 的矩阵
        freqs = torch.outer(t, freqs).float()       #计算位置编码频率矩阵  m * base^(dim/2)



        # 假设 freqs = [x, y]
        # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) #拼接得到最终的cosine和sine矩阵
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) 

        return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed