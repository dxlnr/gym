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

causal_conv1d_fn, causal_conv1d_update = None, None

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


class SelectiveScanFn:
  def __call___(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    pass

  @staticmethod
  def backward(ctx, dout, *args):
    pass

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,return_last_state=False):
  return SelectiveScanFn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


class Mamba:
  def __init__(self,d_model=16,d_state=16,d_conv=4,expand=2,dt_rank="auto",dt_min=0.001,dt_max=0.1,dt_init="random",dt_scale=1.0,dt_init_floor=1e-4,conv_bias=True,bias=False,use_fast_path=True,layer_idx=None,device=None,dtype=None):
    self.d_state = d_state
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

  def __call__(self, x, inference_params=None):
    batch, seqlen, dim = x.shape
    conv_state, ssm_state = None, None
    if inference_params is not None:
      conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
      if inference_params.seqlen_offset > 0:
        out, _, _ = self.step(x, conv_state, ssm_state)
        return out

    xz = (self.in_proj @ x.reshape((dim,batch*seqlen))).reshape((batch,dim,seqlen))
    # We do matmul and transpose BLH -> HBL at the same time
    # xz = rearrange(
    #         self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
    #         "d (b l) -> b d l",
    #         l=seqlen,
    #     )
    # if self.in_proj.bias is not None: xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    if self.in_proj.bias is not None: xz = xz + self.in_proj.bias.to(dtype=xz.dtype).reshape((dim,1))

    # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    A = -self.A_log.exp()  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
    if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
      out = mamba_inner_fn(xz,self.conv1d.weight,self.conv1d.bias,self.x_proj.weight,self.dt_proj.weight,self.out_proj.weight,self.out_proj.bias,A,None,None,self.D,delta_bias=self.dt_proj.bias,delta_softplus=True)
    else:
      x, z = xz.chunk(2, dim=1)
      # Compute short convolution
      if conv_state is not None:
        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        # conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        conv_state.copy_(x.pad((self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
          x = self.act(self.conv1d(x)[..., :seqlen])
        else:
          assert self.activation in ["silu", "swish"]
          # x = causal_conv1d_fn(x=x,weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),bias=self.conv1d.bias,activation=self.activation)
          x = causal_conv1d_fn(x=x,weight=self.conv1d.weight.squeeze(),bias=self.conv1d.bias,activation=self.activation)

          # We're careful here about the layout, to avoid extra transposes.
          # We want dt to have d as the slowest moving dimension
          # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
          # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

          x_dbl = self.x_proj(x.reshape((batch*seqlen, dim)))  # (bl d)
          dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
          dt = self.dt_proj.weight @ dt.t()
          # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
          # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
          # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
          dt = dt.reshape((batch,dim,seqlen))
          B = B.reshape((batch,self.d_state,seqlen)).contiguous()
          C = C.reshape((batch,self.d_state,seqlen)).contiguous()
          assert self.activation in ["silu", "swish"]
          y = selective_scan_fn(x,dt,A,B,C,self.D.float(),z=z,delta_bias=self.dt_proj.bias.float(),delta_softplus=True,return_last_state=ssm_state is not None)
          if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
          y = y.reshape((batch,seqlen,dim))
          out = self.out_proj(y)
      return out

  def step(self):
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
