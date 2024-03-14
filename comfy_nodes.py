import numpy as np
import cv2
import torch

from .utils.blend_screentone import fgbg_hist_matching, blend_screentone, multiply

MAX_RESOLUTION = 8192

class BlendScreentone:

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": { 
                        "colored": ("IMAGE",),
                        "screentone": ("IMAGE",),
                        "cluster": ("INT", {"default": 15}),
                        "screentone_scale": ("FLOAT", {"default": 0.75, "step": 0.01}),
                        "color_scale": ("FLOAT", {"default": 0.25, "step": 0.01}),
                        "scale_by_region": ("BOOLEAN", {"default": True}),
                    },
                    "optional":
                    {
                        "sketch": ("IMAGE", {"default": None}),
                    }
                }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "generate"

    CATEGORY = "image/postprocessing"

    def generate(self, colored: torch.Tensor, screentone, cluster=15, screentone_scale=0.8, color_scale=0.25, scale_by_region=True, sketch = None):
        # b h w c
        device, dtype = colored.device, colored.dtype

        if sketch is None:
            sketch = [None] * colored.shape[0]

        colored_batch, screentone_batch = colored, screentone
        screentone_colorized_list, screentone_final_list = [], []
        for colored, screentone, sk in zip(colored_batch, screentone_batch, sketch):
            colored = 255. * colored.cpu().numpy()
            colored = np.clip(colored, 0, 255).astype(np.uint8)
            screentone = 255. * screentone.cpu().numpy()
            screentone = np.clip(screentone, 0, 255).astype(np.uint8)
            h, w = screentone.shape[:2]
            if colored.shape[0] != h or colored.shape[1] != w:
                colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_CUBIC)

            screentone_colorized, layers, layers_vis = blend_screentone(colored, screentone, seed=0, cluster_n=cluster, scale_by_region=scale_by_region)

            screentone = screentone.mean(axis=2, keepdims=True)
            screentone = screentone_scale * screentone + color_scale * cv2.cvtColor(colored, cv2.COLOR_RGB2GRAY)[..., None]

            screentone = np.clip(screentone, 0, 255).astype(np.uint8)
            sc = cv2.cvtColor(screentone_colorized, cv2.COLOR_RGB2GRAY)
            sc_list = [sc[..., None]]
            fgbg_hist_matching(sc_list, screentone)
            screentone_final = sc_list[0][..., 0]
            if sk is not None:
                sk = sk.cpu().numpy()[..., 0] * 255
                screentone_final = multiply(screentone_final, sk)

            screentone_colorized_list.append(screentone_colorized)
            screentone_final_list.append(screentone_final)

        screentone_final = torch.from_numpy(np.array(screentone_final_list)).to(device=device, dtype=dtype) / 255.
        screentone_colorized = torch.from_numpy(np.array(screentone_colorized_list)).to(device=device, dtype=dtype) / 255.
        return screentone_final, screentone_colorized
    

class EmptyLatentImageAdvanced:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": { 
                        "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                        "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                        "long_side": ("INT", {"default": -1, "min": -1, "max": MAX_RESOLUTION, "step": 8}),
                        "aspect_ratio": ("FLOAT", {"default": -1}),
                    },
                    "optional": {
                        "image": ("IMAGE",),
                    }
                }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1, long_side=-1, aspect_ratio=-1., image=None, ):
        if image is not None and long_side < 0:
            batch_size = image.shape[0]
            height, width = image.shape[1], image.shape[2]

        if long_side > 0:
            if image is not None:
                h, w = image.shape[1], image.shape[2]
                aspect_ratio = w / h
            if aspect_ratio > 0:
                if aspect_ratio > 1:
                    width = long_side
                    height = int(round(long_side / aspect_ratio))
                else:
                    height = long_side
                    width = int(round(long_side * aspect_ratio))

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )





# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageAdvanced": EmptyLatentImageAdvanced,
    "BlendScreentone": BlendScreentone
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # "Example": "Example Node"
}
