#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg

import tensorflow as tf
import time

import tensorvision
import tensorvision.utils as utils


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    FN,FP = np.zeros(thresh.shape),np.zeros(thresh.shape)
    posNum, negNum = 0,0

    colors = np.array(hypes['colors'])

    # for key in hypes['colors']:
    #     colors.append(np.array(hypes['colors'][key]))

    valid_gt = np.all(gt_image == colors[0], axis=2)
    for i in range(1,len(colors)) :
        valid_gt = valid_gt + np.all(gt_image == colors[i], axis=2)

    for i in range(len(colors)) :
        N, P, pos, neg = seg.evalExp(np.all(gt_image == colors[i], axis=2),
                                             cnn_image,
                                             thresh, validMap=None,
                                             validArea=valid_gt)
        FN = np.add(FN,N)
        FP = np.add(FP,P)
        posNum+=pos
        negNum+=neg

    return FN, FP, posNum, negNum


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    num_classes = hypes['arch']['num_classes']
    colors = np.array(hypes['colors'])

    # for i,key in enumerate(colors):
    #     colors[i] = np.array(colors[key])
    #     del colors[key]

    eval_dict = {}
    for phase in ['train','val']:
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_posnum = 0
        total_negnum = 0

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                    datum = datum.rstrip().rstrip('\r')
                    image_file, gt_file = datum.split(" ")
                    image_file = os.path.join(image_dir, image_file)
                    gt_file = os.path.join(image_dir, gt_file)

                    image = scp.misc.imread(image_file, mode='RGB')
                    gt_image = scp.misc.imread(gt_file, mode='RGB')

                    shape = gt_image.shape
                    gt = np.all(gt_image == colors[0], axis=2).reshape(shape[0], shape[1],1)
                    # print (gt.shape)
                    for i in range(1,len(colors)) :

                        new_gt = np.all(gt_image == colors[i], axis=2).reshape(shape[0], shape[1],1)
                        # print(new_gt.shape)
                        gt = np.concatenate((gt, new_gt), axis=2)

                    print (gt.shape)

                    input_image = image

                    shape = input_image.shape

                    feed_dict = {image_pl: input_image}

                    output = sess.run([softmax], feed_dict=feed_dict)
                    # output = np.array(output)

                    # print("Image shape : {}".format(output.shape))
                    # output_im = output.reshape(shape[0], shape[1],num_classes)
                    output = np.argmax(output,axis=2)
                    # raw_output_up = tf.argmax(raw_output_up, dimension=3)
                    # if hypes['jitter']['fix_shape']:
                    #     gt_shape = gt_image.shape
                    #     output_im = output_im[offset_x:offset_x+gt_shape[0],
                    #                           offset_y:offset_y+gt_shape[1]]

                    if phase == 'val':

                        # pred_flatten = tf.reshape(output_im, [-1,])
                        # raw_gt = tf.reshape(gt_image, [-1,])
                        # indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes-1)), 1)
                        # gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                        # pred = tf.gather(pred_flatten, indices)

                        # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

                        name = os.path.basename(image_file)
                        fname = name.split('.')[0]+".txt"


                        print (fname)
                        np.savetxt("testing/"+fname,output,header=image_file,comments="#"+gt_file)

                        # decode_labels
                        inf_image = seg.paint(output.reshape(shape[:2]),colors)

                        infname = "testing/"+name.split('.')[0] + '_color.png'

                        scp.misc.imsave(infname,inf_image)
                        # # green_image = seg.paint(output_im,colors)
                        # image_list.append((name, green_image))


                        # ov_image = seg.blend_transparent(image,green_image)
                        # image_list.append((name2, ov_image))


                    FN, FP, posNum, negNum = eval_image(hypes,
                                                        gt_image, output_im)

                    total_fp += FP
                    total_fn += FN
                    total_posnum += posNum
                    total_negnum += negNum

        # print("Got total FP FN etc\n")
        # eval_dict[phase] = seg.pxEval_maximizeFMeasure(
        #     total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        # if phase == 'val':
        #     start_time = time.time()
        #     for i in xrange(10):
        #         sess.run([softmax], feed_dict=feed_dict)
        #     dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['train']:
        eval_list.append(('[{}] MaxF1'.format(phase),
                          100*eval_dict[phase]['MaxF']))
        eval_list.append(('[{}] BestThresh'.format(phase),
                          100*eval_dict[phase]['BestThresh']))
        eval_list.append(('[{}] Average Precision'.format(phase),
                          100*eval_dict[phase]['AvgPrec']))
    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
