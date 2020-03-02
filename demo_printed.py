#!/usr/bin/env python

import os, sys
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from tqdm import tqdm

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

# To output results in PAGE XML format (http://www.primaresearch.org/schema/PAGE/gts/pagecontent/2013-07-15/)
PAGE_XML_DIR = './page_xml'


def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, kernel_size=5)
    return mask


if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    model_dir = sys.argv[1]
    filename = sys.argv[2]
    outfile = sys.argv[3]

    with tf.Session():  # Start a tensorflow session
      # Load the model
      m = LoadedModel(model_dir, predict_mode='filename')

      # For each image, predict each pixel's label
      prediction_outputs = m.predict(filename)
      probs = prediction_outputs['probs'][0]
      original_shape = prediction_outputs['original_shape']
      probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
      probs = probs / np.max(probs)  # Normalize to be in [0, 1]

      # Binarize the predictions
      page_bin = page_make_binary_mask(probs)

      # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
      bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False)*255,
                                tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)
      cv2.imwrite(outfile, bin_upscaled)
