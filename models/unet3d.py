from tinygrad.nn.state import get_state_dict
from extra.models.unet3d import UNet3D

if __name__ == "__main__":
    mdl = UNet3D()
    mdl.load_from_pretrained()

    for idx, (k, v) in enumerate(get_state_dict(mdl).items()):
        print(f'{idx:5}  {k:40} {str(v.shape):24} {str(v.device):12} {v.dtype}')
