import os
from os.path import join as opj
from glob import glob

import cv2
import json
import numpy as np
import xmltodict
from PIL import Image

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

import torch
from torch.utils.data import Dataset

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

def seg2bbox(seg):
    #Takes in 2d ndarray and returns bboxes of maximum connected area per image
    if seg.sum() == 0:
        return [0,0,0,0]
    py, px = np.where(seg == 1)
    return np.array([px.min(), py.min(), px.max(), py.max()])

class WeakImageNetTrainDataset(Dataset):
    def __init__(self, transforms=None):
        super().__init__()

        self._ROOT = '/home/mtchiu/data/ILSVRC/ILSVRC2012'
        self._ADDON_ROOT = '/home/mtchiu/data/ILSVRC_addon/ILSVRC2012'
        self._INSTANCE_COUNT = 5

        self.transforms = transforms

        with open(opj(self._ADDON_ROOT, 'nid_to_synset_id.json'), 'r') as f:
            self.nid2sid = json.load(f)

        with open(opj(self._ADDON_ROOT, '1000nids.json'), 'r') as f:
            self.child_labels = json.load(f)

        with open(opj(self._ADDON_ROOT, '40supernids.json'), 'r') as f:
            self.super_labels = json.load(f)

        with open(opj(self._ADDON_ROOT, 'seg_label_random10_si.json'), 'r') as f:
            seg_maps = json.load(f)

        self.imgs = sorted(glob(opj(self._ROOT, 'train', '*/*')))
        self.locs = [None] * len(self.imgs)
        self.nids = [None] * len(self.imgs)
        for i_idx in range(len(self.imgs)):
            nid = self.imgs[i_idx].split(os.sep)[-2]
            cls_name = self.imgs[i_idx].split(os.sep)[-1]
            seg_name = cls_name.replace('JPEG', 'png')

            self.nids[i_idx] = nid
            if nid in seg_maps and seg_name in seg_maps[nid]:
                self.locs[i_idx] = opj(self._ADDON_ROOT, 'seg_label', nid, seg_name)
        self.imgs = np.array(self.imgs)
        self.locs = np.array(self.locs)
        self.nids = np.array(self.nids)

        #Only get images with segmentations
        valid_segs = (self.locs != None)
        self.imgs = self.imgs[valid_segs]
        self.locs = self.locs[valid_segs]
        self.nids = self.nids[valid_segs]

        #Make dictionary of images of the same category for each image
        self.segs = {}
        for k in seg_maps:
            self.segs[k] = np.where(self.nids==k)[0]

        print('WeakImageNet - Train - loaded %d local maps' % len(self.imgs))

    def _stitch_image(self, base_idx, base_img, base_nid, base_loc):
        h, w = base_img.shape[:2]

        #Find base img obj for size ratio selection
        base_bbox = seg2bbox(base_loc)
        base_h = base_bbox[3] - base_bbox[1] + 1
        base_w = base_bbox[2] - base_bbox[0] + 1

        #Create canvas for stitching
        stitched_img = base_img
        stitched_loc = [base_loc]
        stitched_box = [base_bbox]

        #Find max num_instances inversely proportional to base obj size
        proportion = base_loc.sum() / (h * w)
        max_instance = max(int(self._INSTANCE_COUNT + 1 - 5*proportion), 1)
        if max_instance == 1:
            total_instance_count = 1
        else:
            total_instance_count = np.random.randint(2, max_instance+1)

        #Shuffle crop img idx to be stitched
        crop_idxs = self.segs[base_nid]
        crop_idxs = crop_idxs[crop_idxs != base_idx]
        crop_idxs = np.random.permutation(crop_idxs)

        instance_count = 1
        for crop_img_idx in crop_idxs:
            if instance_count == total_instance_count: break

            #Read image to be cropped and resize everything to crop_img size
            crop_img = np.array(Image.open(self.imgs[crop_img_idx]).convert('RGB'))
            crop_loc = np.array(Image.open(self.locs[crop_img_idx]))
            crop_loc = cv2.resize(crop_loc, crop_img.shape[1::-1], cv2.INTER_NEAREST)

            #Get obj bbox
            crop_bbox = seg2bbox(crop_loc)
            crop_h = crop_bbox[3] - crop_bbox[1] + 1
            crop_w = crop_bbox[2] - crop_bbox[0] + 1
            #Get obj in crop img
            crop_img = crop_img[crop_bbox[1]:crop_bbox[3]+1, crop_bbox[0]:crop_bbox[2]+1]
            crop_loc = crop_loc[crop_bbox[1]:crop_bbox[3]+1, crop_bbox[0]:crop_bbox[2]+1]

            for _ in range(5):
                #Scale limit, 0.25-0.5x base image, 0.5-2x crop obj
                scale = min(h/crop_h, w/crop_w) * np.random.uniform(0.25, 0.5)
                scale = min(scale, 2)
                i_crop_h = int(crop_h * scale)
                i_crop_w = int(crop_w * scale)

                if i_crop_h <= 0 or i_crop_w <= 0:
                    print('i_crop_h/w = 0', self.imgs[base_idx], self.imgs[crop_img_idx],
                          scale, crop_h, crop_w, base_h, base_w)
                    continue

                #Resize src img and find object bbox for copying
                i_crop_img = cv2.resize(crop_img, (i_crop_w, i_crop_h))
                i_crop_loc = cv2.resize(crop_loc, (i_crop_w, i_crop_h), interpolation=cv2.INTER_NEAREST)

                #Select stitching position on base_img
                px = np.random.randint(w-i_crop_w+1)
                py = np.random.randint(h-i_crop_h+1)
                i_crop_box = np.array([px, py, px+i_crop_w-1, py+i_crop_h-1])

                stch_loc = sum([loc[py:py+i_crop_h, px:px+i_crop_w] for loc in stitched_loc]).astype(np.uint8)

                base_area = stch_loc.sum()
                overlap_area = (i_crop_loc & stch_loc).sum()
                if base_area == 0 or overlap_area / base_area < 0.5:
                    #Stitch obj
                    stitched_img[py:py+i_crop_h, px:px+i_crop_w][i_crop_loc==1] = i_crop_img[i_crop_loc==1]

                    loc = np.zeros(stitched_img.shape[:2], np.uint8)
                    loc[py:py+i_crop_h, px:px+i_crop_w] = i_crop_loc

                    stitched_loc.append(loc)
                    stitched_box.append(i_crop_box)

                    instance_count += 1
                    break

        stitched_loc = np.concatenate([loc[None] for loc in stitched_loc], axis=0)
        return stitched_img, stitched_box, stitched_loc

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(Image.open(self.imgs[index]).convert('RGB'))
        loc = np.array(Image.open(self.locs[index]))
        loc = cv2.resize(loc, img.shape[1::-1], cv2.INTER_NEAREST)

        nid = self.imgs[index].split(os.sep)[-2]
        #Try out contiguous_id w/ bg
        lbl = self.child_labels['nid_to_labelid'][nid] + 1

        img, box, loc = self._stitch_image(index, img, nid, loc)
        img = Image.fromarray(img)
        lbl = torch.tensor([lbl] * len(box))
        loc = torch.tensor(loc)
        loc = SegmentationMask(loc, img.size, mode='mask')

        boxlist = BoxList(box, img.size, mode='xyxy')
        boxlist.add_field('labels', lbl)
        boxlist.add_field('masks', loc)

        if self.transforms is not None:
            img, boxlist = self.transforms(img, boxlist)

        return img, boxlist, index

    def get_img_info(self, index):
        return {'height': 1, 'width': 1}

class WeakImageNetValDataset(Dataset):
    def __init__(self, transforms=None):
        super().__init__()

        self._ROOT = '/home/mtchiu/data/ILSVRC/ILSVRC2012'
        self._ADDON_ROOT = '/home/mtchiu/data/ILSVRC_addon/ILSVRC2012'

        self.transforms = transforms

        with open(opj(self._ADDON_ROOT, 'nid_to_synset_id.json'), 'r') as f:
            self.nid2sid = json.load(f)

        with open(opj(self._ADDON_ROOT, '1000nids.json'), 'r') as f:
            self.child_labels = json.load(f)

        with open(opj(self._ADDON_ROOT, '40supernids.json'), 'r') as f:
            self.super_labels = json.load(f)

        with open(opj(self._ADDON_ROOT, 'seg_label_random10_si.json'), 'r') as f:
            seg_maps = json.load(f)

        self.imgs = sorted(glob(opj(self._ROOT, 'val', '*/*')))
        self.locs = [None] * len(self.imgs)
        self.nids = [None] * len(self.imgs)
        for i in range(len(self.imgs)):
            cls_name = self.imgs[i].split(os.sep)[-1]
            xml_name = cls_name.replace('.JPEG', '.xml')
            xml_path = opj(self._ADDON_ROOT, 'CLS-LOC', 'val', xml_name)

            if os.path.isfile(xml_path):
                with open(xml_path, 'r') as f:
                    xml = xmltodict.parse(f.read())

                h = int(xml['annotation']['size']['height'])
                w = int(xml['annotation']['size']['width'])
                self.locs[i] = {'filename': xml_name, 'size': (h, w), 'bboxes': []}

                objects = xml['annotation']['object']
                if not isinstance(objects, list):
                    objects = [objects]
                for obj in objects:
                    bbox = np.array([
                        float(obj['bndbox']['xmin']),
                        float(obj['bndbox']['ymin']),
                        float(obj['bndbox']['xmax']),
                        float(obj['bndbox']['ymax'])
                    ])
                    self.locs[i]['bboxes'].append(bbox)
        self.locs = np.array(self.locs)
        self.nids = np.array(self.nids)

        print('RSANet_ILSVRC Dataset - Val - loaded %d xmls' % np.sum(self.locs != None))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(Image.open(self.imgs[index]).convert('RGB'))

        nid = self.imgs[index].split(os.sep)[-2]

        h, w = img.shape[:2]
        ah, aw = self.locs[index]['size']
        sh, sw = h/ah, w/aw

        img = Image.fromarray(img)

        box = []
        for bbox in self.locs[index]['bboxes']:
            bbox[[0,2]] *= sw
            bbox[[1,3]] *= sh
            bbox = bbox.astype(int)
            box.append(bbox)

        #Try out contiguous_id w/ bg
        lbl = self.child_labels['nid_to_labelid'][nid] + 1
        lbl = torch.tensor([lbl] * len(box))

        boxlist = BoxList(box, img.size, mode='xyxy')
        boxlist.add_field('labels', lbl)

        if self.transforms is not None:
            img, boxlist = self.transforms(img, boxlist)

        return img, boxlist, index

    def get_img_info(self, index):
        return {'height': 1, 'width': 1}

def WeakImageNetDataset(split, transforms=None):
    if split == 'train':
        return WeakImageNetTrainDataset(transforms)
    else:
        return WeakImageNetValDataset(transforms)
