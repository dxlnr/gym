"""Train a unet3D"""
from pathlib import Path
import os
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
from engine.dataloader import get_batch_load
from engine.conf import Conf
from examples.mlperf.metrics import dice_ce_loss, get_dice_score_np
from extra.lr_scheduler import MultiStepLR
from extra.models.unet3d import UNet3D
from data.kits19 import get_batch, get_data_split, sliding_window_inference
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.device import Device
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import (get_state_dict, load_state_dict, safe_load,
                               safe_save)
from tinygrad.tensor import Tensor, dtypes
from tqdm import tqdm, trange


def train_unet3d():
  conf = Conf()
  GPUS = getenv("GPUS", 0)
  ftx,fty,fvx,fvy = get_data_split(path="/home/daniel/code/datasets/kits19/processed")
  # steps = len(ftx)//(conf.batch_size*GPUS)
  steps = len(ftx)//(conf.batch_size)
  if getenv("WANDB"): import wandb; wandb.init(project="tinygrad-unet3d")

  def get_data(iterator, rank=0):
    # device = f"gpu:{rank}"
    device = Device.DEFAULT if rank == 0 else f"gpu:{rank}"
    x,y = next(iterator)
    return x.to(device).realize(), y.to(device).realize()

  class Trainer:
    def __init__(self, conf, rank=-1):
      self.conf, self.rank = conf, rank
      device = None if rank == -1 or (rank == 0 and GPUS == 1) else f"gpu:{rank}"
      Tensor.manual_seed(1337)
      self.mdl = UNet3D()
      if getenv("PRETRAINED"): self.mdl.load_from_pretrained()
      if getenv("CONTINUE"): load_state_dict(self.mdl, safe_load(os.path.join(conf.load_ckpt_path, f"/tmp/unet3d-ckpt-{conf.start_epoch}.safetensors")))
      self.params = get_state_dict(self.mdl)
      for x in self.params.values() if device else []: x.to_(device)
      self.optim = SGD(self.params.values(), lr=conf.lr, momentum=conf.momentum, nesterov=True, weight_decay=conf.weight_decay)
      if conf.lr_decay_epochs: self.scheduler = MultiStepLR(self.optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)
    def __call__(self, x:Tensor, y:Tensor) -> Tensor:
      self.optim.zero_grad()
      with Tensor.train():
        out = self.mdl(x)
        loss = dice_ce_loss(out, y).backward()
        del out, x, y
      return loss.realize()
    def lr_warmup(self, current_epoch):
      scale = current_epoch / self.conf.lr_warmup_epochs
      self.optim.lr.assign(Tensor([self.conf.init_lr + (self.conf.lr - self.conf.init_lr) * scale])).realize()
    def step(self):
      self.optim.step()
      if self.conf.lr_decay_epochs: self.scheduler.step()
    def eval(self, loader, score_fn=get_dice_score_np, epoch=0, steps=42):
      s = 0
      for i, _ in enumerate(t:=trange(steps)):
        GlobalCounters.reset()
        st = time.perf_counter()
        try:
          x,y = next(loader)
          out, label = sliding_window_inference(self.mdl,x,y)
          s += score_fn(out, label.squeeze(axis=1)).mean(axis=0)
          del out, label
          et = (time.perf_counter()-st)*1000
          t.set_description(f"loss: {(cl/(i+1)):.2f}% step, dice {(s/(i+1)):.2f}: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
          if getenv("WANDB"): wandb.log({"loss": (cl/(i+1)), "step_time_ms": et})
        except StopIteration: break
      return {"epoch": epoch, "mean_dice": s/(i+1)}

  trainers = [Trainer(conf)]
  @TinyJit
  def train(x:Tensor, y:Tensor):
    outs = [trainers[0](x,y)]
    if GPUS == 2: raise NotImplementedError("write real allreduce")
    elif GPUS > 2: raise NotImplementedError("write real allreduce")
    for t in trainers: t.step()
    return outs

  def eval(vt, epoch, rank=0):
    if rank == 0: m=trainers[0].eval(vt, epoch=epoch, steps=len(fvx)//conf.batch_size)
    elif rank == 2: raise NotImplementedError("write real allreduce")
    elif GPUS > 2: raise NotImplementedError("write real allreduce")
    Tensor.training = True
    return m

  is_successful, diverged = False, False
  vl = get_batch(fvx,fvy,batch_size=1, patch_size=conf.val_input_shape, shuffle=False, augment=True)
  for epoch in range(conf.start_epoch+1, conf.epochs):
    if epoch % conf.save_every_epoch == 0: safe_save(get_state_dict(trainers[0].mdl), f"/tmp/unet3d-ckpt-{epoch}.safetensors")
    tl = get_batch(ftx,fty,batch_size=conf.batch_size, patch_size=conf.input_shape, shuffle=True, augment=True)
    # tl = get_batch_load(batch_size=conf.batch_size, patch_size=conf.input_shape, oversampling=conf.oversampling)
    cl = 0
    for i, _ in enumerate(t:=trange(steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      try:
        x,y = next(tl)
        x,y = Tensor(x,dtype=dtypes.float).to(Device.DEFAULT).realize(), Tensor(y,dtype=dtypes.uint8).to(Device.DEFAULT).realize()
        outs = train(x,y)
        cl += sum([loss.item() for loss in outs])/len(outs)
        et = (time.perf_counter()-st)*1000
        t.set_description(f"epoch: {epoch+1} loss: {(cl/(i+1)):.2f}% step: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
        if getenv("WANDB"): wandb.log({"loss": (cl/(i+1)), "step_time_ms": et})
        del x, y
      except StopIteration: break
    del tl

    if epoch % conf.eval_every == 0:
      eval_metrics = eval(vl, epoch=epoch)
      if eval_metrics["mean_dice"] >= conf.quality_threshold:
        print("\nsuccess", eval_metrics["mean_dice"], ">", conf.quality_threshold)
        is_successful = True
      elif eval_metrics["mean_dice"] < 1e-6:
        print("\nmodel diverged. exit.", eval_metrics["mean_dice"], "<", 1e-6)
        diverged = True
    if is_successful or diverged: break

if __name__ == "__main__":
  if not getenv("DIST"):
    train_unet3d()
  else:
    if getenv("CUDA"):
      pass
    else:
      from tinygrad.runtime.ops_gpu import CL
      devices = [f"gpu:{i}" for i in range(len(CL.devices))]
    world_size = len(devices)

