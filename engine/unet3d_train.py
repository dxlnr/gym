"""Train a unet3D"""
import os
import time
import wandb
from dataloader import get_batch_loader
from examples.mlperf.metrics import dice_ce_loss
from extra.lr_scheduler import MultiStepLR
from extra.models.unet3d import UNet3D
from tinygrad.helpers import GlobalCounters, dtypes, flatten, getenv
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import (get_state_dict, load_state_dict, safe_load,
                               safe_save)
from tinygrad.tensor import Tensor
from tqdm import trange


def train_unet3d():
  from examples.mlperf.conf import Conf
  conf = Conf()
  GPUS = getenv("GPUS", 1)
  # steps = len(get_train_files())//(BS*GPUS)
  steps = 168//(conf.batch_size*GPUS)
  if getenv("WANDB"): wandb.init(project="tinygrad-unet3d")

  def data_get(iterator, rank=0):
    device = f"gpu:{rank}"
    x,y,c = next(iterator)
    return x.to(device).realize(), Tensor(y, dtype=dtypes.int32, device=device), c

  class Trainer:
    def __init__(self, conf, rank=-1):
      self.conf, self.rank = conf, rank
      device = None if rank == -1 or (rank == 0 and GPUS == 1) else f"gpu:{rank}"
      Tensor.manual_seed(1337)
      self.mdl = UNet3D()
      if getenv("PRETRAINED"): self.mdl.load_from_pretrained()
      if getenv("CONTINUE"): load_state_dict(self.mdl, safe_load(os.path.join(conf.load_ckpt_path, f"unet3d-ckpt-{conf.start_epoch}.safetensors")))
      self.params = get_state_dict(self.mdl)
      for x in self.params.values() if device else []: x.to_(device)
      self.optim = SGD(self.params.values(), lr=conf.lr, momentum=conf.momentum, nesterov=True, weight_decay=conf.weight_decay)
      if conf.lr_decay_epochs: scheduler = MultiStepLR(self.optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)
      def __call__(self, x:Tensor, y:Tensor) -> Tensor:
        self.optim.zero_grad()
        with Tensor.train():
          out = self.mdl(x)
          loss = dice_ce_loss(out, y).backward()
          del out, x, y
        return loss.realize()
    # def lr_warmup(optim, init_lr, lr, current_epoch, warmup_epochs):
    #   scale = current_epoch / warmup_epochs
    #   optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale])).realize()
    def step(self):
      self.optim.step()
      if self.conf.lr_decay_epochs: self.lr_schedule.step()

  trainers = [Trainer(rank) for rank in range(GPUS)]
  @TinyJit
  def train(*tensors):
    outs = [trainers[i](x,y) for i,(x,y) in enumerate(zip(tensors[::2], tensors[1::2]))]
    # allreduce
    if GPUS == 2:
      raise NotImplementedError("write real allreduce")
    elif GPUS > 2:
      raise NotImplementedError("write real allreduce")
    for t in trainers: t.step()
    return outs

  for epoch in range(conf.epochs):
    if epoch % conf.save_every_epoch == 0: safe_save(get_state_dict(trainers[0].mdl), f"/tmp/unet3d-ckpt-{epoch}.safetensors")
    tl = get_batch_loader(conf.batch_size, conf.input_shape, conf.oversampling)
    proc = [data_get(tl, rank) for rank in range(GPUS)]
    for _ in (t:=trange(steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      outs = train(*flatten([(x,y) for x,y,_ in proc]))
      out_items = [loss.item() for loss in outs]
      loss = sum([x[0] for x in out_items])/len(out_items)
      et = (time.perf_counter()-st)*1000
      t.set_description(f"loss: {loss:.2f}% step: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
      if getenv("WANDB"): wandb.log({"loss": loss, "step_time_ms": et})
