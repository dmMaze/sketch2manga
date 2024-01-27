import os.path as osp
import random
from typing import List, Tuple

import numpy as np
import cv2
import torch
from einops import rearrange
from skimage import color
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from PIL import Image


def get_template_histvq(template: np.ndarray) -> Tuple[List[np.ndarray]]:
    len_shape = len(template.shape)
    num_c = 3
    mask = None
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and template.shape[-1] == 1:
        num_c = 1
    elif len_shape == 3 and template.shape[-1] == 4:
        mask = np.where(template[..., -1])
        template = template[..., :num_c][mask]

    values, quantiles = [], []
    for ii in range(num_c):
        v, c = np.unique(template[..., ii].ravel(), return_counts=True)
        q = np.cumsum(c).astype(np.float64)
        if len(q) < 1:
            return None, None
        q /= q[-1]
        values.append(v)
        quantiles.append(q)
    return values, quantiles


def inplace_hist_matching(img: np.ndarray, tv: List[np.ndarray], tq: List[np.ndarray]) -> None:
    len_shape = len(img.shape)
    num_c = 3
    mask = None

    tgtimg = img
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and img.shape[-1] == 1:
        num_c = 1
    elif len_shape == 3 and img.shape[-1] == 4:
        mask = np.where(img[..., -1])
        tgtimg = img[..., :num_c][mask]

    im_h, im_w = img.shape[:2]
    oldtype = img.dtype
    for ii in range(num_c):
        _, bin_idx, s_counts = np.unique(tgtimg[..., ii].ravel(), return_inverse=True,
                                                return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        if len(s_quantiles) == 0:
            return
        s_quantiles /= s_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, tq[ii], tv[ii]).astype(oldtype)
        if mask is not None:
            img[..., ii][mask] = interp_t_values[bin_idx]
        else:
            img[..., ii] = interp_t_values[bin_idx].reshape((im_h, im_w))


def fgbg_hist_matching(fg_list: List, bg: np.ndarray, min_tq_num=0):
    btv, btq = get_template_histvq(bg)
    idx_matched = -1

    if btq is not None:
        tv, tq = btv, btq
        idx_matched = -1
        
        if len(tq[0]) > min_tq_num:
            for ii, fg in enumerate(fg_list):
                if ii != idx_matched and len(tq[0]) > min_tq_num:
                    inplace_hist_matching(fg, tv, tq)


def skimage_rgb2lab(rgb):
    return color.rgb2lab(rgb.reshape(1,1,3))


def calc_ciede(mean_list, cls_list):
  cls_no = []
  tgt_no = []
  ciede_list = []
  for i in range(len(mean_list)):
    img_1 = np.array(mean_list[i][:3])
    for j in range(len(mean_list)):
      if i == j:
        continue
      img_2 = np.array(mean_list[j][:3])
      ciede = color.deltaE_ciede2000(skimage_rgb2lab(img_1), skimage_rgb2lab(img_2))[0][0]
      ciede_list.append(ciede)
      cls_no.append(cls_list[i])
      tgt_no.append(cls_list[j])
  ciede_df = pd.DataFrame({"cls_no": cls_no, "tgt_no": tgt_no, "ciede2000": ciede_list})
  return ciede_df


def get_cls_update(ciede_df, threshold, cls2counts):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df['ciede2000'] < threshold][['cls_no', 'tgt_no']].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        max_cls = max(merge, key=cls2counts.get)
        for cls in merge:
            if cls != max_cls:
                merge_dict[cls] = max_cls
    return merge_dict


def get_blur_torch(img: torch.Tensor, labels: torch.Tensor, size, blur=True):
    if blur:
        assert size % 2 == 1
        p = (size - 1) // 2
        img = F.pad(img, [p, p, p, p], mode='reflect')
        img = F.avg_pool2d(img, kernel_size=size, stride=1)
    
    cls = torch.unique(labels).reshape(-1, 1, 1, 1)
    masks = torch.bitwise_and(img[:, [3]] > 127, cls == labels)

    cls_counts = masks.sum(dim=(2, 3), keepdim=True) + 1e-7
    rgb_means = (img[:, :3] * masks).sum(dim=(2, 3), keepdim=True) / cls_counts
    
    rgb_means = rgb_means.squeeze().cpu().tolist()
    cls_list = cls.squeeze().cpu().tolist()
    cls_counts = cls_counts.squeeze().cpu().tolist()
    
    return rgb_means, cls_list, cls_counts, masks


def layer_divide_torch(
        img: np.ndarray, 
        loop: int = 3, 
        cls_num: int = 10, 
        threshold: int = 15, 
        size: int = 5, 
        kmeans_samples: int = -1, 
        device: str = 'cpu'):
    """
    See https://github.com/mattyamonaca/layerdivider also https://github.com/mattyamonaca/layerdivider/pull/43

    """
    
    rgb_flatten = cluster_samples = img[..., :3].reshape((-1, 3))
    im_h, im_w = img.shape[:2]

    if img.shape[2] == 4:
        alpha_mask = np.where(img[..., 3] > 127)
    else:
        alpha_mask = np.ones_like(img[..., 0])
        img = np.concatenate([img, alpha_mask[..., None] * 255], axis=2)
        alpha_mask = alpha_mask > 0
        # img = np.concatenate([img, alpha_mask[]])

    resampled = False
    if rgb_flatten.shape[0] > len(alpha_mask[0]):
        cluster_samples = img[..., :3][alpha_mask].reshape((-1, 3))
        resampled = True

    if len(rgb_flatten) > kmeans_samples and kmeans_samples > 0:
        cluster_samples = shuffle(cluster_samples, random_state=0, n_samples=kmeans_samples)
        resampled = True

    kmeans = MiniBatchKMeans(n_clusters=cls_num, n_init='auto').fit(cluster_samples)
    if resampled:
        labels = kmeans.predict(rgb_flatten)
    else:
        labels = kmeans.labels_

    img_torch = rearrange([img], 'n h w c -> n c h w')
    img_torch = torch.from_numpy(img_torch).to(dtype=torch.float32, device=device)
    labels_torch = torch.from_numpy(labels.reshape((1, 1, im_h, im_w))).to(dtype=torch.float32, device=device)

    assert loop > 0
    img_torch_ori = img_torch.clone()
    for i in range(loop):
        rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size)
        ciede_df = calc_ciede(rgb_means, cls_list)
        cls2rgb, cls2counts, cls2masks = {}, {}, {}
        for c, rgb, count, mask in zip(cls_list, rgb_means, cls_counts, masks):
            cls2rgb[c] = rgb
            cls2counts[c] = count
            cls2masks[c] = mask[None, ...]
        merge_dict = get_cls_update(ciede_df, threshold, cls2counts)
        tgt2merge, notmerged = {}, set(cls_list)
        for k, v in merge_dict.items():
            if v not in tgt2merge:
                tgt2merge[v] = []
            tgt2merge[v].append(k)
            notmerged.remove(k)
        for k in notmerged:
            tgt2merge[k] = []

        for tgtc, srcc_list in tgt2merge.items():
            mask = cls2masks[tgtc]
            for srcc in srcc_list:
                mask = torch.bitwise_or(mask, cls2masks[srcc])
            labels_torch.masked_fill_(mask, tgtc)
            if i != loop - 1:
                for jj in range(3):
                    img_torch[:, jj].masked_fill_(mask[0], cls2rgb[tgtc][jj])

    cls_list = torch.unique(labels_torch)
    img_torch = img_torch_ori
    rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size, blur=False)
    for mask, rgb in zip(masks, rgb_means):
        for jj in range(3):
            img_torch[:, jj][mask] = rgb[jj]
    
    img = rearrange(img_torch.cpu().to(torch.float32).numpy(), 'n c h w -> h w (n c)')
    img = img.clip(0, 255).astype(np.uint8)
    labels = labels_torch.cpu().to(torch.float32).numpy().squeeze().astype(np.uint32)
    return img, labels


def minmax_transform(img: np.ndarray):
    _min, _max = img.min(), img.max()
    _span = _max - _min
    if _span == 0:
        _span += 1e-6
    img = (img - _min) / _span * 255
    return img


def multiply(img: np.ndarray, target: np.ndarray):
    if target.shape[0] != img.shape[0] or target.shape[1] != img.shape[1]:
        target = Image.fromarray(target).resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS)
        target = np.array(target)

    intype = img.dtype
    img = img * target.astype(np.float32)
    
    scaler = 255 / (img.max() - img.min() + 1e-9)
    img = np.clip(img * scaler, 0, 255).astype(intype)
    return img


def gamma_transform(img: np.ndarray, gamma: float, normalize=True):
    img = (img.astype(np.float32) / 255.) ** gamma

    if normalize:
        img = minmax_transform(img).astype(np.uint8)

    return img
    

def blend_screentone(colorized: np.ndarray, screened: np.ndarray, cluster_n=10, tone_std_tol=5, vis_layers: bool = False, seed=None, scale_by_region=True):

    def _blend_pixels(hsv, tgt_tone, rst, layermask):

        pixels_hsv = hsv[layermask][:, None, :]

        tones = tgt_tone[layermask]
        nhist, vals = np.histogram(tones)

        kmin = vals[0]
        kmax = vals[-1]
        thr = tones.shape[0] * 0.3
        for ii, val in enumerate(nhist.cumsum()):
            if val > thr:
                kmin = vals[ii]
                break

        for ii, val in enumerate(nhist[::-1].cumsum()):
            if val > thr:
                kmax = vals[len(vals) - ii - 1]
                break

        span = kmax - kmin
        tones = np.clip(tones, kmin, kmax)

        tone_std = np.std(tones)

        tone_std = np.clip(tone_std, 0, tone_std_tol)
        scaler1 = 1 + 0.8 * tone_std / tone_std_tol
        scaler2 = 1 - 0.4 * tone_std / tone_std_tol


        base1 = (kmax - tones) / span
        base2 = 1 - base1

        pixels_s = pixels_hsv[..., 1]
        pixels_hsv[..., 1] = pixels_s * (base1 * scaler1 + base2 * scaler2)
        pixels_hsv = np.clip(pixels_hsv,0,255).astype(np.uint8)

        pixels_rgb = cv2.cvtColor(pixels_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)[:, 0]

        rst[layermask] = pixels_rgb
        return rst
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    screentone_colorized = np.zeros_like(colorized)
    colorized_hsv = cv2.cvtColor(colorized, cv2.COLOR_RGB2HSV).astype(np.float32)
    # colorized_hsv = cv2.cvtColor(colorized, cv2.COLOR_RGB2HLS).astype(np.float32)

    if len(screened.shape) == 3:
        if screened.shape[2] > 1:
            screened = screened.mean(axis=2, keepdims=True)

    if not scale_by_region:
        layermask = np.full((colorized.shape[0], colorized.shape[1]), True)
        # screentone_colorized = _blend_pixels(colorized_hsv, screened, screentone_colorized, layermask)
        # return screentone_colorized, None, None
        screened = screened[..., 0].astype(np.float32)
        screened = (screened.max() - screened) / (screened.max() - screened.min())

        colorized_hsv[..., 1] = colorized_hsv[..., 1] * (screened * 3 + (1-screened) * 0.6)
        colorized_hsv = np.clip(colorized_hsv,0,255).astype(np.uint8)

        screentone_colorized = cv2.cvtColor(colorized_hsv, cv2.COLOR_HSV2RGB)
        # screentone_colorized = cv2.cvtColor(colorized_hsv, cv2.COLOR_HLS2RGB)
        return screentone_colorized, None, None
    
    layers, labels = layer_divide_torch(np.array(colorized), cls_num=cluster_n)
    layers = layers[..., :3]
    layers_vis = layers.copy()

    for ii in range(cluster_n):
        glabels = labels == ii
        label_counts = glabels.sum()
        if label_counts == 0:
            continue
        label_mask = np.array(glabels, dtype=np.uint8) * 255

        num_labels, local_labels, stats, centroids = cv2.connectedComponentsWithStats(label_mask)

        blended_mask = np.zeros_like(glabels)
        for jj in range(num_labels):
            if jj == 0:
                continue
            nl = stats[jj, 4]
            if nl / num_labels < 0.1:
                continue
            layermask = local_labels == jj
            if layermask.sum() == 0:
                continue
            _blend_pixels(colorized_hsv, screened, screentone_colorized, layermask)

            blended_mask = np.bitwise_or(blended_mask, layermask)

        layermask = np.bitwise_and(glabels, np.bitwise_not(blended_mask))
        if layermask.sum() == 0:
            continue

        _blend_pixels(colorized_hsv, screened, screentone_colorized, layermask)

    return screentone_colorized, layers, layers_vis