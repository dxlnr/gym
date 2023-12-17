"""Dataloading"""
import random, time
import numpy as np
from tinygrad.helpers import dtypes, prod
from tinygrad.tensor import Tensor
from multiprocessing import Queue, Process, shared_memory

def get_batch_load(batch_size=2, val=False, shuffle=True, patch_size=(128,128,128), augment=True, oversampling=0.4, nw=4):
  from pathlib import Path
  import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
  from data.kits19 import get_data_split, transform
  if val:
    _,_, fx,fy = get_data_split(path="/home/daniel/code/datasets/kits19/processed")
  else: 
    fx,fy,_,_ = get_data_split(path="/home/daniel/code/datasets/kits19/processed")

  sz = (len(fx),1,*patch_size)
  try:
    shmX,shmY = shared_memory.SharedMemory(name="kits19_X"),shared_memory.SharedMemory(name="kits19_Y")
  except FileNotFoundError:
    shmX,shmY = shared_memory.SharedMemory(name="kits19_X", create=True, size=prod(sz)), shared_memory.SharedMemory(name="kits19_Y", create=True, size=prod(sz))

  X = Tensor.empty(*sz, dtype=dtypes.float, device="disk:/dev/shm/kits19_X")
  Y = Tensor.empty(*sz, dtype=dtypes.uint8, device="disk:/dev/shm/kits19_Y")

  q_in, q_out = Queue(), Queue()

  def loader_process(q_in, q_out, X:Tensor, Y:Tensor):
    while (_recv := q_in.get()) is not None:
      idx, xfn, yfn = _recv
      vol,label = np.load(xfn), np.load(yfn)
      if augment: vol,label = transform(vol,label,patch_size,oversampling)
      X[idx].assign(vol.tobytes())
      Y[idx].assign(label.tobytes())
      q_out.put(idx)

  procs = []
  for _ in range(nw):
    p = Process(target=loader_process, args=(q_in, q_out, X, Y))
    p.daemon = True
    p.start()
    procs.append(p)

  gen = iter(random.sample(list(range(len(fx))),len(fx))) if shuffle else iter(range(len(fx)))
  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      i = next(gen)
      xfn, yfn = fx[i], fy[i]
      q_in.put((idx, xfn, yfn))
  for bn in range(len(fx)//batch_size): enqueue_batch(bn)

  gotten = [0]*(len(fx)//batch_size)
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size]

  for _ in range(len(fx)//batch_size): yield receive_batch()
  # shutdown processes
  for _ in procs: q_in.put(None)
  for p in procs: p.join()
  shmX.close(); shmY.close()
  shmX.unlink(); shmY.unlink()

def dist_batch(fx, yx, batch_size=2, patch_size=(128,128,128), shuffle=True, augment=True, oversampling=0.4, nw=4):
  q_in, q_out = Queue(), Queue()
  procs = []
  for _ in range(nw):
    p = Process(target=get_batch, args=(q_in, q_out, fx, fy))
    p.daemon = True
    p.start()
    procs.append(p)

if __name__ == "__main__":
  THREADS = 4
  BS = 2
  PATCH_SIZE = (128,128,128)
  DEVICE = "cpu"
  st = time.monotonic()
  iterator = get_batch_load(batch_size=BS, val=False, patch_size=PATCH_SIZE, nw=THREADS)
  while True:
    try:
      x,y = next(iterator)
      x.to(DEVICE).realize(), y.to(DEVICE).realize()
      del x,y
    except StopIteration: break
  print(f"CPU | workers: {THREADS:2d} | runtime {time.monotonic()-st:.2f}s | batch size {BS} | sz {PATCH_SIZE}")

  from pathlib import Path
  import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
  from data.kits19 import get_data_split, get_batch
  fx,fy,_,_ = get_data_split(path="/home/daniel/code/datasets/kits19/processed")

  st = time.monotonic()
  iterator = get_batch(fx,fy,batch_size=BS, patch_size=PATCH_SIZE, shuffle=True, augment=True)
  while True:
    try:
      x,y = next(iterator)
      Tensor(x,dtype=dtypes.float).to(DEVICE).realize(), Tensor(y,dtype=dtypes.uint8).to(DEVICE).realize()
      del x,y
    except StopIteration: break
  print(f"CPU | workers: 1 | runtime {time.monotonic()-st:.2f}s | batch size {BS} | sz {PATCH_SIZE}")
