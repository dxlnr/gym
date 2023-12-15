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
from tinygrad.device import Device
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
  # steps = 168//(conf.batch_size)
  steps = 2
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
      if getenv("CONTINUE"): load_state_dict(self.mdl, safe_load(os.path.join(conf.load_ckpt_path, f"unet3d-ckpt-{conf.start_epoch}.safetensors")))
      self.params = get_state_dict(self.mdl)
      for x in self.params.values() if device else []: x.to_(device)
      self.optim = SGD(self.params.values(), lr=conf.lr, momentum=conf.momentum, nesterov=True, weight_decay=conf.weight_decay)
      if conf.lr_decay_epochs: self.scheduler = MultiStepLR(self.optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)
    def __call__(self, x:Tensor, y:Tensor) -> Tensor:
      self.optim.zero_grad()
      with Tensor.train():
        print("x", x.shape, "y", y.shape)
        out = self.mdl(x)
        loss = dice_ce_loss(out, y).backward()
        del out, x, y
      return loss.realize()
    # def lr_warmup(optim, init_lr, lr, current_epoch, warmup_epochs):
    #   scale = current_epoch / warmup_epochs
    #   optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale])).realize()
    def step(self):
      self.optim.step()
      if self.conf.lr_decay_epochs: self.scheduler.step()
    def eval(self, epoch):
      # if epoch == next_eval_at:
      # next_eval_at += conf.eval_every
        # dtype_im = dtypes.half if getenv("FP16") else dtypes.float

      eval_metrics = evaluate(conf, mdl, val_loader, epoch=epoch)
      Tensor.training = True
        # print("  (eval):", eval_metrics)

        # safe_save(get_state_dict(mdl), os.path.join(conf.save_ckpt_path, f"unet3d-ckpt-{epoch}.safetensors"))
      if eval_metrics["mean_dice"] >= conf.quality_threshold:
        print("\nsuccess", eval_metrics["mean_dice"], ">", conf.quality_threshold, "runtime", time.monotonic()-t0_total)
        safe_save(get_state_dict(mdl), os.path.join(self.conf.save_ckpt_path, f"unet3d-ckpt-{epoch}.safetensors"))
        is_successful = True
        elif eval_metrics["mean_dice"] < 1e-6:
          print("\nmodel diverged. exit.", eval_metrics["mean_dice"], "<", 1e-6)
          diverged = True

      if is_successful or diverged:
        break

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
    # cl = []
    # proc = [data_get(tl, rank) for rank in range(GPUS)]
    # proc = [data_get(tl)]
    cl = 0
    for i, _ in enumerate(t:=trange(steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      x,y = get_data(tl)
      # outs = train(*flatten([(x,y) for x,y,_ in proc]))
      outs = train(x,y)
      # out_items = [loss.item() for loss in outs]
      # loss = sum(out_items)/len(out_items)
      cl += sum([loss.item() for loss in outs])/len(outs)
      et = (time.perf_counter()-st)*1000
      t.set_description(f"loss: {(cl/(i+1)):.2f}% step: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
      if getenv("WANDB"): wandb.log({"loss": (cl/(i+1)), "step_time_ms": et})

if __name__ == "__main__":
    train_unet3d()
