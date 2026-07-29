# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import math
import random

import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    # fields = ["labels", "area", "iscrowd"]
    fields = ["labels", "area"]
    if "pseudo_mask" in target.keys(): # for pseudo labels
        fields.append("pseudo_mask")
    if "weight" in target.keys(): # for pseudo labels
        fields.append("weight")
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    if max_size is not None: # size有可能出现max_size+1的情况，额外添加
        size = tuple(min(s, max_size) for s in size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def random_ratio_resize(image, target, image_scale,ratio_range=(0.1, 2.0)):
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale
    def rescale_size(old_size,scale):
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                            max_short_edge / min(h, w))
        else:
            raise TypeError(
                f'Scale must be a number or tuple of int, but got {type(scale)}')

        new_size = _scale_size((w, h), scale_factor)
        return new_size
    def _scale_size(size,scale,):
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        w, h = size
        return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)
    scale = random_sample_ratio(image_scale, ratio_range)
    size = rescale_size(image.size,scale)
    rescaled_image = F.resize(image, size)
    if target is None:
        return rescaled_image, None
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
    return rescaled_image, target

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomRatioResize(object):
    def __init__(self, img_scale=(896, 896), ratio_range=(0.1, 2.0)):
        assert isinstance(img_scale, (list, tuple))
        self.img_scale = img_scale
        self.ratio_range = ratio_range
    def __call__(self, img, target=None):
        return random_ratio_resize(img, target, self.img_scale,self.ratio_range)

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ToRGB(object):
    def __call__(self, img, target):
        return img.convert("RGB"), target

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


def _box_iou_np(box, boxes):
    """IoU of one xyxy box [4] against an [N,4] xyxy array. Returns [N]."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a0 = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    a1 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.clip(a0 + a1 - inter, 1e-6, None)


class CopyPasteShrink(object):
    """Copy-paste-shrink augmentation: inject synthetic SMALL pseudo-boxes.

    Runs AFTER the geometric transforms and BEFORE Normalize, so the image is a
    PIL image at final network resolution and boxes are xyxy ABSOLUTE pixels.
    It copies an existing pseudo-box crop, shrinks it to a sampled target area
    (in that post-resize pixel space), pastes it at a low-IoU location, and
    registers the paste as a new pseudo-box (pseudo_mask=1) so it flows through
    the exact same downstream path (DTQT/DN, matching, pseudo loss) as the real
    pseudo-boxes. Only inserted into the pipeline when cps_enable=True; the stock
    path never constructs it.

    Pasted boxes inherit their source pseudo-box's label (-1). All pseudo-boxes
    already share label -1 (category_id=-1 in OW_COCO_R2.json), so no downstream
    consumer requires per-row label uniqueness.
    """

    def __init__(self, num_pastes=2, prob=1.0, target_area_min=200.0,
                 target_area_max=1024.0, source_max_side=80.0, feather_px=3,
                 weight_scale=1.0, max_downscale=6.0, max_place_tries=50):
        self.num_pastes = num_pastes
        self.prob = prob
        self.target_area_min = float(target_area_min)
        self.target_area_max = float(target_area_max)
        self.source_max_side = float(source_max_side)
        self.feather_px = int(feather_px)
        self.weight_scale = float(weight_scale)
        self.max_downscale = float(max_downscale)
        self.max_place_tries = int(max_place_tries)
        self.stats = None  # optional dict populated by the smoke harness

    def _feather_alpha(self, h, w):
        if self.feather_px <= 0:
            return np.ones((h, w, 1), dtype=np.float32)
        yy = np.minimum(np.arange(h), np.arange(h)[::-1]).reshape(h, 1)
        xx = np.minimum(np.arange(w), np.arange(w)[::-1]).reshape(1, w)
        d = np.minimum(yy, xx).astype(np.float32)  # distance to nearest edge
        sigma = self.feather_px / 2.0
        alpha = 1.0 - np.exp(-(d * d) / (2.0 * sigma * sigma))
        return alpha.reshape(h, w, 1).astype(np.float32)

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target
        if "boxes" not in target or "pseudo_mask" not in target:
            return image, target
        boxes = target["boxes"]
        pmask = target["pseudo_mask"].to(torch.bool)
        if boxes.numel() == 0 or int(pmask.sum()) == 0:
            return image, target

        boxes_np = boxes.numpy()
        W, H = image.size
        rgb = image.convert("RGB")
        img_np = np.asarray(rgb).astype(np.float32)

        pmask_np = pmask.numpy()
        pseudo_boxes = boxes_np[pmask_np]
        pseudo_weights = target["weight"][pmask].numpy()
        sides = np.maximum(pseudo_boxes[:, 2] - pseudo_boxes[:, 0],
                           pseudo_boxes[:, 3] - pseudo_boxes[:, 1])
        qualified = np.nonzero(sides <= self.source_max_side)[0]
        smallest = int(np.argmin(sides))  # fallback pool: smallest available

        forbidden = list(boxes_np)  # all GT + pseudo boxes, xyxy-abs
        pasted_boxes, pasted_weights = [], []
        attempted = placed = fallback = 0

        for _ in range(self.num_pastes):
            attempted += 1
            if len(qualified) > 0:
                si = int(random.choice(qualified))
            else:
                si = smallest
                fallback += 1
            sx1, sy1, sx2, sy2 = pseudo_boxes[si]
            sw, sh = sx2 - sx1, sy2 - sy1
            if sw < 1 or sh < 1:
                continue
            target_area = random.uniform(self.target_area_min, self.target_area_max)
            f = math.sqrt(target_area / (sw * sh))
            f = min(f, 1.0)  # never upscale; we want small pastes
            if f < 1.0 / self.max_downscale:
                continue  # downscale cap prevents reaching the target -> skip
            nw = max(1, int(round(sw * f)))
            nh = max(1, int(round(sh * f)))
            cx1, cy1 = int(round(max(0, sx1))), int(round(max(0, sy1)))
            cx2, cy2 = int(round(min(W, sx2))), int(round(min(H, sy2)))
            if cx2 - cx1 < 1 or cy2 - cy1 < 1:
                continue
            crop = rgb.crop((cx1, cy1, cx2, cy2)).resize((nw, nh), Image.Resampling.LANCZOS)
            crop_np = np.asarray(crop).astype(np.float32)
            alpha = self._feather_alpha(nh, nw)

            cand = None
            for _try in range(self.max_place_tries):
                if W - nw <= 0 or H - nh <= 0:
                    break
                px, py = random.randint(0, W - nw), random.randint(0, H - nh)
                c = np.array([px, py, px + nw, py + nh], dtype=np.float32)
                ious = _box_iou_np(c, np.asarray(forbidden, dtype=np.float32))
                if ious.size == 0 or ious.max() < 0.2:
                    cand = c
                    break
            if cand is None:
                continue

            px, py = int(cand[0]), int(cand[1])
            region = img_np[py:py + nh, px:px + nw, :]
            img_np[py:py + nh, px:px + nw, :] = region * (1.0 - alpha) + crop_np * alpha
            forbidden.append(cand)
            pasted_boxes.append([float(cand[0]), float(cand[1]), float(cand[2]), float(cand[3])])
            pasted_weights.append(float(pseudo_weights[si]) * self.weight_scale)
            placed += 1

        if self.stats is not None:
            self.stats["images"] += 1
            self.stats["attempted"] += attempted
            self.stats["placed"] += placed
            self.stats["fallback"] += fallback
            self.stats["areas"].extend(
                (b[2] - b[0]) * (b[3] - b[1]) for b in pasted_boxes
            )

        if placed == 0:
            return image, target

        out_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode="RGB")
        pb = torch.as_tensor(pasted_boxes, dtype=boxes.dtype)
        n = len(pasted_boxes)
        target = target.copy()
        target["boxes"] = torch.cat([boxes, pb], dim=0)
        target["labels"] = torch.cat(
            [target["labels"], torch.full((n,), -1, dtype=target["labels"].dtype)]
        )
        target["pseudo_mask"] = torch.cat(
            [target["pseudo_mask"], torch.ones(n, dtype=target["pseudo_mask"].dtype)]
        )
        target["weight"] = torch.cat(
            [target["weight"], torch.as_tensor(pasted_weights, dtype=target["weight"].dtype)]
        )
        if "area" in target:
            pa = torch.as_tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in pasted_boxes],
                                 dtype=target["area"].dtype)
            target["area"] = torch.cat([target["area"], pa])
        return out_img, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
