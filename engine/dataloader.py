import random, time, ctypes, struct
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from tinygrad.helpers import dtypes, getenv, prod, Timing
from tinygrad.tensor import Tensor
from multiprocessing import Queue, Process, shared_memory, connection, Lock


def get_batch_load(batch_size=2, val=False, shuffle=True, patch_size=(128,128,128), augment=True, oversampling=0.4, nw=8):
  from pathlib import Path
  import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
  from data.kits19 import get_data_split, transform
  tx,ty,_,_ = get_data_split(path="/home/daniel/code/datasets/kits19/processed")

  # sz = (batch_size,1,*patch_size)
  sz = (len(tx),1,*patch_size)
  shmX = shared_memory.SharedMemory(name="kits19_X", create=True, size=prod(sz))
  shmY = shared_memory.SharedMemory(name="kits19_Y", create=True, size=prod(sz))
  X = Tensor.empty(*sz, dtype=dtypes.float, device=f"disk:/dev/shm/kits19_X")
  Y = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/kits19_Y")

  q_in, q_out = Queue(), Queue()

  def loader_process(q_in, q_out, X:Tensor, Y:Tensor):
    while (_recv := q_in.get()) is not None:
      idx, xfn, yfn = _recv
      vol,label = np.load(xfn), np.load(yfn)
      if augment:vol,label = transform(vol,label,patch_size, oversampling)
      X[idx].assign(vol.tobytes())
      Y[idx].assign(label.tobytes())
      q_out.put(idx)

  procs = []
  for _ in range(nw):
    p = Process(target=loader_process, args=(q_in, q_out, X, Y))
    p.daemon = True
    p.start()
    procs.append(p)

  gen = iter(random.sample(list(range(len(tx))),len(tx))) if shuffle else iter(range(len(tx)))
  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      i = next(gen)
      xfn, yfn = tx[i], ty[i]
      q_in.put((idx, xfn, yfn))
  for bn in range(len(tx)//batch_size): enqueue_batch(bn)

  # class Cookie:
  #   def __init__(self, num): self.num = num
  #   def __del__(self):
  #     try: enqueue_batch(self.num)
  #     except StopIteration: pass

  # gotten = [0]*BATCH_COUNT
  gotten = [0]*(len(tx)//batch_size)
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    # return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], Cookie(num)
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size]

  # NOTE: this is batch aligned, last ones are ignored
  for _ in range(0, len(tx)//batch_size): yield receive_batch()

  # shutdown processes
  for _ in procs: q_in.put(None)
  for p in procs: p.join()
  shmX.close(); shmY.close()
  shmX.unlink(); shmY.unlink()


def data_get(iterator, rank=0):
  # device = f"gpu:{rank}"
  device = "CPU"
  x,y = next(iterator)
  return x.to(device).realize(), x.to(device).realize()

if __name__ == "__main__":
  iterator = get_batch_load(batch_size=2, val=False, patch_size=(32,32,32))
  # proc = [data_get(iterator, rank) for rank in range(1)]
  # print(proc)
  # print(len(proc))
  # for i, batch in enumerate(tqdm(train_loader, total=total_batches)):
  while True:
    try: x,y = next(iterator)
    except StopIteration: break
