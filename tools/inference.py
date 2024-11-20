import numpy as np
import os.path as osp
import os
from mmcv import Config, DictAction
from mmdet.models import build_detector
import torch
from PIL import Image
from torchvision import transforms
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mmdet.datasets.pipelines import Compose
import argparse
from tqdm import tqdm
import cv2

def compute_boundary_acc(gt, seg, mask):
    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps
    mask_acc = [None] * num_steps
    bnd_regions = []

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (curr_radius*2+1, curr_radius*2+1))
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        bnd_regions.append(boundary_region)

        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1-gt_in_bound) * (1-mask_in_bound)).sum()

        seg_acc[i] = num_seg_gd_pix / num_edge_pixels
        mask_acc[i] = num_mask_gd_pix / num_edge_pixels

    return sum(seg_acc)/num_steps, sum(mask_acc)/num_steps, bnd_regions

def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)
    
    return intersection, union

def cal_miou(all_gts, all_refines, all_coarses):

    total_new_i = 0
    total_new_u = 0
    total_old_i = 0
    total_old_u = 0

    total_num_images = 0
    total_seg_acc = 0
    total_mask_acc = 0

    all_gts

    good_cases = []
    bad_cases = []

    
    for idx,(gt_path,seg_path,mask_path) in enumerate(tqdm(zip(all_gts,all_coarses,all_refines),total=len(all_gts))):
        gt = np.array(Image.open(gt_path).convert('L'))

        seg = np.array(Image.open(seg_path).convert('L'))

        mask = np.array(Image.open(mask_path).convert('L'))

        """
        Compute IoU and boundary accuracy
        """
        gt = gt >= 128
        seg = seg >= 128
        mask = mask >= 128

        old_i, old_u = get_iu(gt, seg)
        new_i, new_u = get_iu(gt, mask)

        total_new_i += new_i
        total_new_u += new_u
        total_old_i += old_i
        total_old_u += old_u

        seg_acc, mask_acc, bnd_regions = compute_boundary_acc(gt, seg, mask)
        total_seg_acc += seg_acc
        total_mask_acc += mask_acc
        total_num_images += 1

        if mask_acc > seg_acc:
            good_cases.append(dict(
                gt_file = gt_path,
                old_acc = seg_acc,
                new_acc = mask_acc,
                old_iou = old_i / old_u,
                new_iou = new_i / new_u
            ))
        else:
            bad_cases.append(dict(
                gt_file = gt_path,
                old_acc = seg_acc,
                new_acc = mask_acc,
                old_iou = old_i / old_u,
                new_iou = new_i / new_u
            ))

    new_iou = total_new_i/total_new_u
    old_iou = total_old_i/total_old_u
    new_mba = total_mask_acc/total_num_images
    old_mba = total_seg_acc/total_num_images

    print('New IoU  : ', new_iou)
    print('Old IoU  : ', old_iou)
    print('IoU Delta: ', new_iou-old_iou)

    print('New mBA  : ', new_mba)
    print('Old mBA  : ', old_mba)
    print('mBA Delta: ', new_mba-old_mba)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using segrefiner')
    parser.add_argument('--out_dir', default='/data3/zhangsn/Project/liyu/SegRefiner/inference_dir_DeelLab/', type=str, help='Output fine_mask file directory')
    parser.add_argument('--config_dir', default='/data3/zhangsn/Project/liyu/SegRefiner/configs/segrefiner/segrefiner_big.py', type=str, help='Config file directory')
    parser.add_argument('--ckpt_dir', default='/data3/zhangsn/Project/liyu/SegRefiner/work_dirs/liyu002/seg_lr_hr/segrefiner_hr_latest.pth', type=str, help='Ckpt file directory')
    parser.add_argument('--file_gt_dir', default='/data3/zhangsn/Project/liyu/SegRefiner/coarse_data/BIG_DeepLab_MS', type=str, help='GT mask file directory')
    parser.add_argument('--file_dir', default='/data3/zhangsn/Project/liyu/SegRefiner/coarse_data/BIG_DeepLab_MS', type=str, help='Image and Coarse mask file directory')
    parser.add_argument('--mean', default=[123.675, 116.28, 103.53], type=list, help='Images mean')
    parser.add_argument('--std', default=[58.395, 57.12, 57.375], type=list, help='Images std')
    args = parser.parse_args()
    
    # create output path
    os.makedirs(args.out_dir, exist_ok=True)
    
    # build model
    cfg=Config.fromfile(args.config_dir)
    cfg.model.type="SegRefinerSemantic"
    cfg.model.step = 6
    cfg.model.train_cfg = None
    model = build_detector(cfg.model,test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.ckpt_dir, map_location='cpu', strict=True)
    model = build_dp(model, 'cuda', device_ids=[4])
    model.eval()

    # create transform
    img_norm_cfg = dict(
        mean=args.mean, std=args.std, to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadCoarseMasks', test_mode=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'coarse_masks'])
    ]
    pipeline = Compose(test_pipeline)
    
    # collect imgs, coarse masks, gt masks, fine_grained masks
    img_list = []
    coarse_list = []
    gt_list = []
    refine_list = []
    for fname in os.listdir(args.file_dir):
        if fname.split('.')[0].split('_')[-1] == 'im':
            img_list.append(fname)
            continue
        elif fname.split('.')[0].split('_')[-1] == 'seg':
            coarse_list.append(fname)
            continue
        elif fname.split('.')[0].split('_')[-1] == 'gt':
            continue
        elif fname.split('.')[-1] == 'mat':
            continue
        else:
            raise NotImplementedError('The data template is not supported')

    # sorted imgs and coarse masks list
    img_list = sorted(img_list, key=lambda x: x[:-6])
    coarse_list = sorted(coarse_list, key=lambda x: x[:-7])
    for idx,(filename,maskname) in enumerate(tqdm(zip(img_list,coarse_list), total=len(img_list))):
        data_infos = {'filename': filename, 'gtname': None, 'maskname': maskname}
        assert osp.exists(osp.join(args.file_dir,filename)), f"Ori img file does not exist: {osp.join(args.file_dir,filename)}"
        assert osp.exists(osp.join(args.file_dir,maskname)), f"Coarse mask img does not exist: {osp.join(args.file_dir,maskname)}"
        coarse_info=dict(masks = data_infos['maskname'])
        inputs = dict(img_info=data_infos, coarse_info=coarse_info)
        inputs['img_prefix'] = args.file_dir
        inputs['bbox_fields'] = []
        inputs['mask_fields'] = []
        inputs['seg_fields'] = []
        
        # processing input
        inputs = pipeline(inputs)
        
        # predict
        with torch.no_grad():
            img_metas = [inputs['img_metas'].data]
            img = inputs['img'].data.unsqueeze(0).cuda(4)
            coarse_masks = [torch.zeros(1)]
            coarse_masks[0].masks = inputs['coarse_masks'].data.to_ndarray()
            result=model.module.simple_test_semantic(img_metas=img_metas, img=img, coarse_masks=coarse_masks, return_loss=False)[0][0]

        # get finegrained mask
        refine_mask = (result * 255).astype(np.uint8)
        out_name = maskname.split('.')[0]+'refine.png'
        out_file = osp.join(args.out_dir, out_name)
        Image.fromarray(refine_mask).save(out_file)
        refine_list.append(out_name)
    
    # get gt mask
    for fname in os.listdir(args.file_gt_dir):
        if fname.split('.')[0].split('_')[-1] == 'im':
            continue
        elif fname.split('.')[0].split('_')[-1] == 'seg':
            continue
        elif fname.split('.')[0].split('_')[-1] == 'gt':
            gt_list.append(fname)
            continue
        elif fname.split('.')[-1] == 'mat':
            continue
        else:
            raise NotImplementedError('The data template is not supported')
        
    # sorted gt masks and finegrained masks list   
    gt_list = sorted(gt_list, key=lambda x: x[:-6])
    refine_list = sorted(refine_list, key=lambda x: x[:-13])
    
    # collect files path
    gt_paths = [osp.join(args.file_gt_dir,x) for x in gt_list]
    refine_paths = [osp.join(args.out_dir,x) for x in refine_list]
    coarse_paths = [osp.join(args.file_dir,x) for x in coarse_list]
    
    # cal metrics
    metric  = cal_miou(gt_paths, refine_paths, coarse_paths)
