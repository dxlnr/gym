"""Inspect various models"""
import os
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from tinygrad.nn.state import get_state_dict
from models.unet3d import UNet3D
from models.mamba import MambaLMHeadModel

def layers(mdl):
    for idx, (k, v) in enumerate(get_state_dict(mdl).items()):
        print(f'{idx:5}  {k:40} {str(v.shape):24} {str(v.device):12} {v.dtype}')

def model_unet3d():
    mdl = UNet3D()
    mdl.load_from_pretrained()
    layers(mdl)


def model_resnet():
    pass


def model_mamba():
    mdl = MambaLMHeadModel()
    mdl.load_from_pretrained()
    layers(mdl)


if __name__ == "__main__":
    models = os.getenv("MODEL", "resnet,unet3d,mamba").split(",")
    for m in models:
        nm = f"model_{m}"
        if nm in globals():
            print(f"eval {m}")
            globals()[nm]()
            print("")
