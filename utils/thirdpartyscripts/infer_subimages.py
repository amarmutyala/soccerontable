#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
from re import L
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import yaml
import numpy as np
import multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import tqdm
import pycocotools.mask as mask_util


from predictor import VisualizationDemo

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
# constants
WINDOW_NAME = "COCO detections"

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--config-file',
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )

    
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        "--input",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def main(args):
    logger = logging.getLogger(__name__)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if os.path.isdir(args.input):
        im_list = glob.iglob(args.input + '/*.' + args.image_ext)
    else:
        im_list = [args.input]

    for im_name in tqdm.tqdm(im_list, disable=not args.output):
        
            # use PIL, to be consistent with evaluation
            im = read_image(im_name, format="BGR")
            # start_time = time.time()
            # predictions, visualized_output = demo.run_on_image(im)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         im_name,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )

            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(im_name))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     visualized_output.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit

    
            
            out_name = os.path.join(
                args.output, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
            )
            logger.info('Processing {} -> {}'.format(im_name, out_name))
            
            

            # ======================================================================
            h, w = im.shape[:2]

            subimages = []
            for x in range(3):
                for y in range(3):
                    x1, y1 = x*h//4, y*w//4
                    x2, y2 = (x+2)*h//4, (y+2)*w//4
                    subimages.append([x1, y1, x2, y2])

            
     
            out_name_yml = os.path.join(
                args.output, '{}'.format(os.path.basename(im_name)[:-4] + '.yml')
            )

            _mask = np.zeros((h, w), dtype=np.uint8)
            all_boxes = np.zeros((0, 4))
            all_classes = []
            all_segs = []
            for index in range(len(subimages)):
                x1, y1, x2, y2 = subimages[index]

                predictions, visualized_output = demo.run_on_image(im[x1:x2, y1:y2, :])
                preds = predictions['instances'].to('cpu')
                boxes, segms, classes = preds.pred_boxes, preds.pred_masks, preds.pred_classes
                if boxes is None:
                    continue

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output,  str(index) + '_' + os.path.basename(im_name))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)

                # mask_util.encode(np.asfortranarray(segms[i]))
                for i in range(len(boxes)):
                    _tmp = np.zeros((h, w), dtype=np.uint8, order='F')
                    # __segm = mask_util.decode(segms[i])
                    _tmp[x1:x2, y1:y2] = segms[i]
                    __tmp = mask_util.encode(_tmp)
                    all_segs.append(__tmp)

                    _mask[x1:x2, y1:y2] += segms[i].numpy()
                    all_classes.append(classes[i])

                boxes_np = boxes.tensor.numpy()
                boxes_np[:, 0] += y1
                boxes_np[:, 2] += y1
                boxes_np[:, 1] += x1
                boxes_np[:, 3] += x1

                all_boxes = np.vstack((all_boxes, boxes_np))

            _mask = _mask.astype(bool).astype(int)
            out_name_mask = os.path.join(
                args.output, '{}'.format(os.path.basename(im_name)[:-4] + '.png')
            )
            cv2.imwrite(out_name_mask, _mask*255)



            with open(out_name_yml, 'w') as outfile:
                yaml.dump({'boxes': all_boxes,
                        'segms': all_segs,
                        'classes': all_classes}, outfile, default_flow_style=False)


            # logger.info('Saving time: {:.3f}s'.format(time.time() - t))
            # for k, v in timers.items():
            #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        # ======================================================================


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    args = parse_args()
    main(args)
