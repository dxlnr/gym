import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Node
from tinygrad.helpers import dtypes

def one_hot(arr: Tensor, layout="NCDHW", channel_axis=1, num_classes=3):
  arr = arr.squeeze(dim=channel_axis)
  arr= Tensor.eye(num_classes, dtype=dtypes.int32, device=arr.device)[arr]
  if layout == "NCDHW": arr= arr.permute(0, 4, 1, 2, 3)
  return arr

def one_hot_np(arr: np.array, num_classes=3):
  res = np.eye(num_classes)[np.array(arr).reshape(-1)]
  arr = res.reshape(list(arr.shape) + [num_classes])
  arr = arr.transpose((0, 4, 1, 2, 3)).astype(np.float32)
  return arr

def get_dice_score_np(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6):
  channel_axis, reduce_axis = 1, tuple(range(2, len(prediction.shape)))
  prediction = prediction.argmax(axis=channel_axis)
  prediction, target= one_hot_np(prediction)[:, 1:], one_hot_np(target)[:, 1:]
  intersection = np.sum(prediction * target, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return result[0]

def get_dice_score(prediction: Tensor, target: Tensor, prediction_argmax=False, to_onehot_x=False, to_onehot_y=True, layout="NCDHW", smooth_nr=1e-6, smooth_dr=1e-6):
  if layout == "NCDHW":
    channel_axis = 1
    reduce_axis = list(range(2, len(prediction.shape)))
  else:
    channel_axis = -1
    reduce_axis = list(range(1, len(prediction.shape) - 1))
  if prediction_argmax:
    prediction = prediction.argmax(axis=channel_axis)
  else:
    prediction = prediction.softmax(axis=channel_axis)

  if to_onehot_x: prediction = one_hot(prediction, layout=layout, channel_axis=channel_axis)
  if to_onehot_y: target = one_hot(target, layout=layout, channel_axis=channel_axis)

  assert target.shape == prediction.shape, f"Shapes do not match. prediction: {prediction.shape}, target: {target.shape}."
  intersection = (target * prediction).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  return (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)

def cross_entropy_loss(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
  divisor = y.shape[1]
  assert not isinstance(divisor, Node), "sint not supported as divisor"
  y = (1 - label_smoothing)*y + label_smoothing / divisor
  if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
  if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
  return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

def dice_ce_loss(out, label):
  ce = cross_entropy_loss(out, one_hot(label))
  dice_score = get_dice_score(out, label)
  dice = (1. - dice_score).mean()
  return (ce + dice) / 2
