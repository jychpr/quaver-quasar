# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
import copy
import torch
from util.misc import inverse_sigmoid
import torch.nn.functional as F


def prepare_for_cdn_ov(
    dn_args,
    training,
    num_queries,
    num_classes,
    text_embbeding,
    label_enc_embbeding,
    iou_banded_noise=False,
):
    device = text_embbeding.device
    if training:
        # * dn_number 100
        # * label_noise_ratio 0.5
        # * box_noise_scale 1.0
        # positive and negative dn queries
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        targets=targets_preprocess(targets)
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).to(device) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]  # gt sum of each img
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                # 选一个batch中label最多的
                dn_number = dn_number // (int(max(known_num) * 2))  # e.g. 12
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
       
        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)  #  single_pad个box，每个box一个pos一个neg，共有dn_number组
        positive_idx = (
            torch.tensor(range(len(boxes)))
            .long()
            .to(device)
            .unsqueeze(0)
            .repeat(dn_number, 1)
        )
        positive_idx += (
            (torch.tensor(range(dn_number)) * len(boxes) * 2)
            .long()
            .to(device)
            .unsqueeze(1)
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned = known_labels_expaned.scatter(0, chosen_indice, new_label)
        dn_iou_fallback = None
        if box_noise_scale > 0:
            if iou_banded_noise:
                # dn_iou_banded_noise: rejection-sample so each noised box lands in a
                # target IoU band with its source (pos [0.5,1.0), neg (0.0,0.3]).
                known_bbox_expand, n_fb, n_tot = iou_banded_box_noise(
                    known_bboxs, positive_idx, negative_idx, box_noise_scale
                )
                dn_iou_fallback = (n_fb, n_tot)
            else:
                known_bbox_ = torch.zeros_like(known_bboxs)
                known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
                known_bbox_[:, 2:] = (known_bboxs[:, :2] + known_bboxs[:, 2:] / 2)  #  cx,cy,w,h->x,y,x,y
                diff = torch.zeros_like(known_bboxs)
                diff[:, :2] = known_bboxs[:, 2:] / 2  # 宽高除以2
                diff[:, 2:] = known_bboxs[:, 2:] / 2
                rand_sign = (
                    torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)
                    * 2.0
                    - 1.0
                )  #  -1或者1,控制box的noise方向
                rand_part = torch.rand_like(known_bboxs)
                rand_part[negative_idx] += 1.0
                rand_part *= rand_sign
                known_bbox_ = (
                    known_bbox_
                    + torch.mul(rand_part, diff).to(device) * box_noise_scale
                )
                known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
                known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
                known_bbox_expand[:, 2:] = (
                    known_bbox_[:, 2:] - known_bbox_[:, :2]
                )  #  x,y,x,y->cx,cy,w,h
        m = known_labels_expaned.long().to(device)
        m[m==-1]=num_classes-1
        input_label_embed = label_enc_embbeding(m)
        input_label_embed[positive_idx]+=text_embbeding[-1][None,:] # 正样本text embbeding全为object
        input_label_embed[negative_idx]+=text_embbeding[m[negative_idx]] # 负样本可以有noise text embbeding
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_label = torch.zeros(pad_size, 256).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                # [0:14,14:196]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True

            if i == dn_number - 1:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2
                ] = True
            else:
                # [0:14,14:196]
                # [14:28,28:196]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
                # [0:14,:0]
                # [14:28,:14]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i
                ] = True

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
            "pseudo_targets":targets
        }
        if dn_iou_fallback is not None:
            dn_meta["dn_iou_fallback"] = dn_iou_fallback
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
    #  input_query_label -> bs,pad_size,256
    #  input_query_bbox -> bs,pad_size,4
    #  attn_mask -> tgt_size,tgt_size  
    return input_query_label, input_query_bbox, attn_mask, dn_meta

def targets_preprocess(targets):
    new_target=[]
    for target in targets:
        new_target_i=copy.deepcopy(target)
        pseudo_mask=new_target_i['pseudo_mask'].to(torch.bool)
        new_target_i['boxes']=new_target_i['boxes'][pseudo_mask]
        new_target_i['labels']=new_target_i['labels'][pseudo_mask]
        new_target_i['weight']=new_target_i['weight'][pseudo_mask]
        new_target.append(new_target_i)
    # If there is no pseudo annotation, use a real annotation for training
    if sum([len(target['labels']) for target in new_target])==0:
        for i,target in enumerate(targets):
            new_target_i=copy.deepcopy(target)
            if len(new_target_i['labels'])>0:
                new_target_i['boxes']=new_target_i['boxes'][0].unsqueeze(0)
                new_target_i['labels']=new_target_i['labels'][0].unsqueeze(0)
                new_target_i['weight']=new_target_i['weight'][0].unsqueeze(0)
                new_target[i]=new_target_i
                break
    return new_target
        
def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {"pred_logits": output_known_class[-1],
                "pred_boxes": output_known_coord[-1],}
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord


def _diag_iou_xyxy(a, b):
    """Row-wise IoU between two [N,4] xyxy box sets (no autograd needed)."""
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def _stock_box_noise_xyxy(known_bboxs, negative_idx, box_noise_scale):
    """Stock size-proportional corner noise, returned in xyxy (used only as the
    fallback for boxes that miss the IoU band). Mirrors the inline stock scheme."""
    known_bbox_ = torch.zeros_like(known_bboxs)
    known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
    known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
    diff = torch.zeros_like(known_bboxs)
    diff[:, :2] = known_bboxs[:, 2:] / 2
    diff[:, 2:] = known_bboxs[:, 2:] / 2
    rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
    rand_part = torch.rand_like(known_bboxs)
    rand_part[negative_idx] += 1.0
    rand_part *= rand_sign
    known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
    return known_bbox_.clamp(min=0.0, max=1.0)


def iou_banded_box_noise(known_bboxs, positive_idx, negative_idx, box_noise_scale,
                         max_attempts=20):
    """Rejection-sample DN box noise so each noised box lands in a target IoU band
    with its source box: positives in [0.5, 1.0), negatives in (0.0, 0.3]. Boxes
    that miss the band after `max_attempts` fall back to the stock scheme.

    Returns (noised_boxes_cxcywh [N,4], n_fallback, n_total)."""
    device = known_bboxs.device
    N = known_bboxs.shape[0]
    # source boxes -> xyxy, and per-corner half-extents (as in the stock scheme)
    src = torch.zeros_like(known_bboxs)
    src[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
    src[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
    half = known_bboxs[:, 2:] / 2
    diff = torch.cat([half, half], dim=1)  # [N,4]

    is_neg = torch.zeros(N, dtype=torch.bool, device=device)
    is_neg[negative_idx] = True
    # negatives shift harder (base>=1 like stock neg, larger ceiling); positives stay small
    base = torch.where(is_neg, torch.ones(N, device=device), torch.zeros(N, device=device)).unsqueeze(1)
    ceil = torch.where(is_neg, torch.full((N,), 3.0, device=device),
                       torch.full((N,), 0.9, device=device)).unsqueeze(1)

    out_xyxy = src.clone()
    accepted = torch.zeros(N, dtype=torch.bool, device=device)
    for _ in range(max_attempts):
        need = ~accepted
        if not bool(need.any()):
            break
        u = torch.rand(N, 4, device=device)
        sign = torch.randint(0, 2, (N, 4), device=device, dtype=torch.float32) * 2.0 - 1.0
        mag = torch.rand(N, 1, device=device) * ceil
        cand = (src + sign * (base + u) * diff * mag).clamp(0.0, 1.0)
        valid = (cand[:, 2] > cand[:, 0]) & (cand[:, 3] > cand[:, 1])
        iou = _diag_iou_xyxy(cand, src)
        pos_ok = (iou >= 0.5) & (iou < 1.0)
        neg_ok = (iou > 0.0) & (iou <= 0.3)
        in_band = torch.where(is_neg, neg_ok, pos_ok) & valid
        take = need & in_band
        out_xyxy[take] = cand[take]
        accepted |= take

    n_fallback = int((~accepted).sum().item())
    if n_fallback > 0:
        fb = _stock_box_noise_xyxy(known_bboxs, negative_idx, box_noise_scale)
        out_xyxy = torch.where(accepted.unsqueeze(1), out_xyxy, fb)

    res = torch.zeros_like(known_bboxs)
    res[:, :2] = (out_xyxy[:, :2] + out_xyxy[:, 2:]) / 2
    res[:, 2:] = out_xyxy[:, 2:] - out_xyxy[:, :2]
    return res, n_fallback, N
