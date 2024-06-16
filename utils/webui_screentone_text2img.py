import os.path as osp
import os
from pathlib import Path
import io
import json
import base64

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import imageio
from einops import rearrange
from requests.auth import HTTPBasicAuth

from utils.io_utils import find_all_imgs, submit_request, img2b64, save_encoded_image


def long_side_to(H, W, long_side):
    asp = H / W
    if asp > 1:
        H = long_side
        H = int(round(H / 32)) * 32
        W = int(round(H / asp / 32)) * 32
    else:
        W = long_side
        W = int(round(W / 32)) * 32
        H = int(round(W * asp / 32)) * 32
    return H, W


def resize_longside_to(img: Image.Image, long_side) -> Image:
    W, H = img.width, img.height
    H, W = long_side_to(H, W, long_side)
    return img.resize((W, H), resample=Image.Resampling.LANCZOS)


ctrlnet_lineart_cfg = '''
url: ''
mode: invert

long_side: 768
sd_params:
  prompt: ''
  batch_size: 1
  negative_prompt: (bad-artist:1.0), (worst quality, low quality:1.4), (bad_prompt_version2:0.8), bad-hands-5,lowres, bad anatomy, bad hands, ((text)), (watermark), error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, ((username)), blurry, (extra limbs)
  width: 1024
  height: 576
  steps: 40
  cfg_scale: 9
  seed: 0
  sampler_name: "DPM++ 2M SDE"
  alwayson_scripts: 
    controlnet:
      args: [
        {
            "enabled": True,
            "image": "",
            "module": "invert (from white bg & black line)",
            "model": "control_v11p_sd15_lineart [43d4be0d]",
            "weight": 1,
            "resize_mode": "Inner Fit (Scale to Fit)",
            "low_vram": False,
            # "processor_res": resolution,
            # "threshold_a": 64,
            # "threshold_b": 64,
            # "guidance": 1,
            # "guidance_start": 0,
            "guidance_end": 1,
            "pixel_perfect": True
        }
      ]

'''


def colorize_sketch(sketch, prompt, long_side=1024, cfg=ctrlnet_lineart_cfg, negative_prompt='', seed=0, url: str = None):

    if isinstance(cfg, str):
        if osp.isfile(cfg):
            cfg = OmegaConf.load(cfg)
        else:
            cfg = OmegaConf.create(cfg)
    if url is not None:
        cfg.url = url

    data = {
        **OmegaConf.to_container(cfg.sd_params),
        # "init_images": [img_b64]
    }

    auth = None
    if 'username' in cfg:
        username = cfg.pop('username')
        password = cfg.pop('password')
        auth = HTTPBasicAuth(username, password)

    if isinstance(sketch, np.ndarray):
        sketch = Image.fromarray(sketch).convert('RGB')

    W, H = sketch.width, sketch.height
    H, W = long_side_to(H, W, long_side)
    sketch_resized = resize_longside_to(sketch, long_side)
    data['width'], data['height'] = W, H
    data['prompt'] = prompt
    data['negative_prompt'] = negative_prompt
    data['seed'] = seed

    if cfg.mode == 'invert' or cfg.mode == 'line':
        img_b64 = img2b64(sketch_resized)
        data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
    elif cfg.mode == 'greyscale':
        img_b64 = img2b64(255 - np.array(sketch_resized.convert('L').convert('RGB')))
        data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
    elif cfg.mode == 'invert_greyscale':
        img_b64 = img2b64(np.array(sketch_resized.convert('L').convert('RGB')))
        data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
    else:
        raise

    response = submit_request(cfg.url, json.dumps(data), auth=auth)
    images = response.json()['images']
    rst = [Image.open(io.BytesIO(base64.b64decode(img_b64))) for img_b64 in images]
    return rst


def color2screentone(args, save_results: bool = False, long_side: int = None):
    data = {
        **OmegaConf.to_container(args.sd_params),
    }

    auth = None
    if 'username' in args:
        username = args.pop('username')
        password = args.pop('password')
        auth = HTTPBasicAuth(username, password)

    img_path = args.img_path
    if osp.isfile(img_path):
        imglist = [img_path]
    else:
        imglist = find_all_imgs(img_path, abs_path=True)

    if hasattr(args, 'max_nimgs'):
        maxn = int(args.max_nimgs)
        if maxn < len(imglist):
            imglist = imglist[:maxn]
    
    for ii, img_path in enumerate(imglist):
        if isinstance(img_path, str):
            cimg = Image.open(img_path).convert('RGB')
        elif isinstance(img_path, np.ndarray):
            cimg = Image.fromarray(img_path)
        else:
            assert isinstance(img_path, Image.Image)
            cimg = img_path
        W, H = cimg.width, cimg.height
        if long_side is None:
            long_side = args.long_side
        H, W = long_side_to(H, W, long_side)
        data['width'], data['height'] = W, H
        img_resized = cimg.resize((W, H), resample=Image.Resampling.LANCZOS)

        seed = data['seed']
        
        if args.mode == 'invert':
            img_b64 = img2b64(img_resized)
            data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
        elif args.mode == 'greyscale':
            img_b64 = img2b64(255 - np.array(img_resized.convert('L').convert('RGB')))
            data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
        elif args.mode == 'invert_greyscale':
            img_b64 = img2b64(np.array(img_resized.convert('L').convert('RGB')))
            data['alwayson_scripts']['controlnet']['args'][0]['image'] = img_b64
        else:
            raise

        response = submit_request(args.url, json.dumps(data), auth=auth)
        images = response.json()['images']

        rstlist = [np.array(img_resized)]
        for output_img_b64 in images:
            cimg = Image.open(io.BytesIO(base64.b64decode(output_img_b64)))
            rstlist.append(np.array(cimg))
        
        if save_results:
            if hasattr(args, 'save_dir'):
                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                imname = osp.basename(img_path).replace(Path(img_path).suffix, '') if isinstance(img_path, str) else str(ii)
                imgsavep = osp.join(args.save_dir, f'{imname}_seed{seed}.jpg')

            if hasattr(args, 'save_dir') and args.save_compared:
                img = rearrange(rstlist, 'n h w c -> h (n w) c')
                imageio.imwrite(imgsavep, img, q=100)
            else:
                output_img_b64 = images[0]    
                save_encoded_image(output_img_b64, imgsavep)
        
        return rstlist