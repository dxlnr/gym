"""Train a unet3D"""
from pathlib import Path
import os
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
from engine.dataloader import get_batch_load
from engine.conf import Conf
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
  conf = Conf()
  GPUS = getenv("GPUS", 0)
  # steps = len(get_train_files())//(BS*GPUS)
  # steps = 168//(conf.batch_size*GPUS)
  steps = 168//(conf.batch_size)
  if getenv("WANDB"): import wandb; wandb.init(project="tinygrad-unet3d")

  def get_data(iterator, rank=0):
    # device = f"gpu:{rank}"
    device = "cpu"
    x,y = next(iterator)
    return x.to(device).realize(), y.to(device).realize()

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

  # trainers = [Trainer(conf, rank) for rank in range(GPUS)]
  trainers = [Trainer(conf)]
  @TinyJit
  # def train(*tensors):
  def train(x:Tensor, y:Tensor):
    # outs = [trainers[i](x,y) for i,(x,y) in enumerate(zip(tensors[::2], tensors[1::2]))]
    # outs = [trainers[0](x,y) for i,(x,y)]
    outs = [trainers[0](x,y)]
    # allreduce
    if GPUS == 2:
      raise NotImplementedError("write real allreduce")
    elif GPUS > 2:
      raise NotImplementedError("write real allreduce")
    for t in trainers: t.step()
    return outs

  for epoch in range(conf.epochs):
    if epoch % conf.save_every_epoch == 0: safe_save(get_state_dict(trainers[0].mdl), f"/tmp/unet3d-ckpt-{epoch}.safetensors")
    tl = get_batch_load(batch_size=conf.batch_size, patch_size=conf.input_shape, oversampling=conf.oversampling)
    # proc = [data_get(tl, rank) for rank in range(GPUS)]
    # proc = [data_get(tl)]
    for _ in (t:=trange(steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      x,y = get_data(tl)
      # outs = train(*flatten([(x,y) for x,y,_ in proc]))
      outs = train(x,y)
      out_items = [loss.item() for loss in outs]
      loss = sum([x[0] for x in out_items])/len(out_items)
      et = (time.perf_counter()-st)*1000
      t.set_description(f"loss: {loss:.2f}% step: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
      if getenv("WANDB"): wandb.log({"loss": loss, "step_time_ms": et})

if __name__ == "__main__":
    train_unet3d()
