"""Mamba: Linear-Time Sequence Modeling with Selective State Spaces."""
from typing import Optional
from pathlib import Path
import math
from functools import partial
import torch
from dataclasses import dataclass, field
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, fetch, get_child

@dataclass
class MambaConfig:
  d_model: int = 768
  n_layer: int = 24
  vocab_size: int = 50277
  ssm_cfg: dict = field(default_factory=dict)
  rms_norm: bool = True
  residual_in_fp32: bool = True
  fused_add_norm: bool = True
  pad_vocab_size_multiple: int = 8

class Mamba:
  def __init__(self,d_model=16,d_state=16,d_conv=4,expand=2,dt_rank="auto",dt_min=0.001,dt_max=0.1,dt_init="random",dt_scale=1.0,dt_init_floor=1e-4,conv_bias=True,bias=False,use_fast_path=True,layer_idx=None,device=None,dtype=None):
    self.d_inner = int(expand*d_model)
    self.dt_rank = math.ceil(d_model/16) if dt_rank == "auto" else dt_rank
    self.activation = "silu"

    self.in_proj = nn.Linear(d_model,self.d_inner*2,bias=bias)
    self.conv1d = nn.Conv1d(self.d_inner,self.d_inner,bias=conv_bias,kernel_size=d_conv,groups=self.d_inner,padding=d_conv-1)
    self.x_proj = nn.Linear(self.d_inner,self.dt_rank+d_state*2, bias=False)
    self.dt_proj = nn.Linear(self.dt_rank,self.d_inner,bias=True)

    A = Tensor.arange(1, d_state+1, dtype=dtypes.float).repeat((self.d_inner,1)).contiguous()
    self.A_log = A.log()
    self.D = Tensor.ones(self.d_inner)

    self.out_proj = nn.Linear(self.d_inner,d_model,bias=bias)

  def __call__(self, x):
    pass

class Block:
  def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
    self.residual_in_fp32 = residual_in_fp32
    self.fused_add_norm = fused_add_norm
    self.mixer = mixer_cls(dim)
    self.norm = norm_cls(dim)

  def __call__(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
    pass

def create_block(d_model,ssm_cfg=None,norm_epsilon=1e-5,rms_norm=False,residual_in_fp32=False,fused_add_norm=False,layer_idx=None):
  if ssm_cfg is None: ssm_cfg = {}
  mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
  # norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
  norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)
  block = Block(d_model,mixer_cls,norm_cls=norm_cls,fused_add_norm=fused_add_norm,residual_in_fp32=residual_in_fp32)
  block.layer_idx = layer_idx
  return block

class MixerModel:
  def __init__(self,d_model:int,n_layer:int,vocab_size:int,ssm_cfg=None,norm_epsilon=1e-5,rms_norm=False,initializer_cfg=None,fused_add_norm=False,residual_in_fp32=False) -> None:
    self.residual_in_fp32 = residual_in_fp32
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.layers = [create_block(d_model,ssm_cfg=ssm_cfg,norm_epsilon=norm_epsilon,rms_norm=rms_norm,residual_in_fp32=residual_in_fp32,fused_add_norm=fused_add_norm,layer_idx=i) for i in range(n_layer)]
    self.norm_f = (nn.LayerNorm)(d_model, eps=norm_epsilon)


class MambaLMHeadModel:
  def __init__(self, config=MambaConfig(), initializer_cfg=None):
    self.config = config
    d_model = config.d_model
    n_layer = config.n_layer
    vocab_size = config.vocab_size
    ssm_cfg = config.ssm_cfg
    rms_norm = config.rms_norm
    residual_in_fp32 = config.residual_in_fp32
    fused_add_norm = config.fused_add_norm
    pad_vocab_size_multiple = config.pad_vocab_size_multiple

    if vocab_size % pad_vocab_size_multiple != 0: vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    self.backbone = MixerModel(
      d_model=d_model,
      n_layer=n_layer,
      vocab_size=vocab_size,
      ssm_cfg=ssm_cfg,
      rms_norm=rms_norm,
      initializer_cfg=initializer_cfg,
      fused_add_norm=fused_add_norm,
      residual_in_fp32=residual_in_fp32,
    )
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

  def load_from_pretrained(self):
    fn = Path(__file__).parents[1] / "weights"
    if self.config.d_model == 2560 and self.config.n_layer == 64: mn="mamba-2.8b.ckpt"
    elif self.config.d_model == 2048 and self.config.n_layer == 48: mn="mamba-1.4b.ckpt"
    elif self.config.d_model == 1536 and self.config.n_layer == 48: mn="mamba-790m.ckpt"
    elif self.config.d_model == 1024 and self.config.n_layer == 48: mn="mamba-370m.ckpt"
    elif self.config.d_model == 768 and self.config.n_layer == 24: mn="mamba-130m.ckpt"
    else: raise ValueError(f"Unsupported pretrained config: dim {self.config.d_model}, layers {self.config.n_layer}")
    fn = fn / mn
    fetch(f"https://huggingface.co/state-spaces/{mn.replace('.ckpt','')}/resolve/main/pytorch_model.bin?download=true", fn)
    test = torch.load(fn, map_location=torch.device("cpu"))
    for k, v in test.items():
      obj = get_child(self, k)
      assert obj.shape == v.shape, (k, obj.shape, v.shape)
      obj.assign(v.numpy())

if __name__ == "__main__":
  BATCH, LENGTH, DIM = 2, 64, 16
  m = MambaLMHeadModel()
  m.load_from_pretrained()
