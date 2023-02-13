#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# Name        : create_dataset
#
# Description : Creates a YOLO dataset using MosaicDataSet.
#
# Note        : This program is not particularly optimized or well organized.
#               It requires severe refactoring. Don't like it? Great.
#
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 24-Jan-2023 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################
###############################################################################
# IMPORTS
###############################################################################

import os
import numpy as np
from skimage.io import imsave
from util import put_labels_in_image,create_directory
from mosaicdataset import MosaicDataSet

###############################################################################
# CONSTANT PARAMETERS
###############################################################################

# =============================================================================
# FILE EXTENSIONS
# =============================================================================

# Input data file extensions
EXT_MOSAIC_IMAGE='.png'
EXT_MOSAIC_LABELS='.txt'

# Output data file extensions
EXT_DATASET_IMAGE='.png'
EXT_DATASET_LABELS='.txt'

# =============================================================================
# PATHS
# =============================================================================

# Input mosaics paths. Mosaic image and labels files must be in the same
# directory and have the same name except for the extension. The image must
# have EXT_MOSAIC_IMAGE extension and the labels EXT_MOSAIC_LABELS extension.

PATH_TRAIN_MOSAICS=['../../DATA/BURROWS/MOSAICS/H1C1057_1-001_V2',
                    '../../DATA/BURROWS/MOSAICS/H1C1057_1-004_V2',
                    '../../DATA/BURROWS/MOSAICS/H1C1057_1-007_V2',
                    '../../DATA/BURROWS/MOSAICS/H1C3178_1-001_V2']

PATH_TEST_MOSAICS=['../../DATA/BURROWS/MOSAICS/H1C3178_1-007_V2']

# Output dataset paths
PATH_DATASET_TRAIN_IMG='../../DATA/BURROWS/DATASET/images/train/'
PATH_DATASET_TRAIN_LBL='../../DATA/BURROWS/DATASET/labels/train/'
PATH_DATASET_TRAIN_PVW='../../DATA/BURROWS/DATASET/preview/train/'
PATH_DATASET_VALIDATION_IMG='../../DATA/BURROWS/DATASET/images/validation/'
PATH_DATASET_VALIDATION_LBL='../../DATA/BURROWS/DATASET/labels/validation/'
PATH_DATASET_VALIDATION_PVW='../../DATA/BURROWS/DATASET/preview/validation/'
PATH_DATASET_TEST_IMG='../../DATA/BURROWS/DATASET/images/test/'
PATH_DATASET_TEST_LBL='../../DATA/BURROWS/DATASET/labels/test/'
PATH_DATASET_TEST_PVW='../../DATA/BURROWS/DATASET/preview/test/'

# =============================================================================
# DATASET GENERATION PARAMETERS
# =============================================================================

WIDTH_IMAGE=640
HEIGHT_IMAGE=480
MIN_LABELS_PER_IMAGE=0
MIN_LABEL_AREA=0.5
MAX_SEARCH_ITERATIONS=1000
REJECT_IMAGES_WITH_PARTIAL_LABELS=True

NUM_IMAGES=5000
RATIO_TRAIN=0.8
RATIO_VALIDATION=0.1
RATIO_TEST=0.1

###############################################################################
# MAIN PROGRAM
###############################################################################

# =============================================================================
# INIT
# =============================================================================

numTrainImagePerMosaic=int(round(NUM_IMAGES*RATIO_TRAIN/len(PATH_TRAIN_MOSAICS)))
numValImagesPerMosaic=int(round(NUM_IMAGES*RATIO_VALIDATION/len(PATH_TRAIN_MOSAICS)))
numTestImagesPerMosaic=int(round(NUM_IMAGES*RATIO_TEST/len(PATH_TEST_MOSAICS)))

trainImageNumber=0
validationImageNumber=0
testImageNumber=0

create_directory(PATH_DATASET_TRAIN_IMG,True)
create_directory(PATH_DATASET_TRAIN_LBL,True)
create_directory(PATH_DATASET_TRAIN_PVW,True)
create_directory(PATH_DATASET_VALIDATION_IMG,True)
create_directory(PATH_DATASET_VALIDATION_LBL,True)
create_directory(PATH_DATASET_VALIDATION_PVW,True)
create_directory(PATH_DATASET_TEST_IMG,True)
create_directory(PATH_DATASET_TEST_LBL,True)
create_directory(PATH_DATASET_TEST_PVW,True)

dataSet=MosaicDataSet()

# =============================================================================
# LOOP FOR EACH TRAIN MOSAIC
# =============================================================================

for idxMosaic,curMosaic in enumerate(PATH_TRAIN_MOSAICS):
    print('* PROCESSING MOSAIC %s (%d of %d)'%(curMosaic,idxMosaic+1,len(PATH_TRAIN_MOSAICS)))

    # Create the train dataset
    dataSet.create(mosaicFileName=curMosaic+EXT_MOSAIC_IMAGE,
                    labelsFileName=curMosaic+EXT_MOSAIC_LABELS,
                    imgWidth=WIDTH_IMAGE,
                    imgHeight=HEIGHT_IMAGE,
                    mosaicYRatioMax=RATIO_TRAIN,
                    minArea=MIN_LABEL_AREA,
                    minLabels=MIN_LABELS_PER_IMAGE,
                    maxIter=MAX_SEARCH_ITERATIONS,
                    rejectPartial=REJECT_IMAGES_WITH_PARTIAL_LABELS)

    # Export the train data items
    for idxImage in range(numTrainImagePerMosaic):
        print('  + SAVING TRAIN DATA %d of %d (MOSAIC %d of %d)'%(idxImage+1,numTrainImagePerMosaic,idxMosaic+1,len(PATH_TRAIN_MOSAICS)))
        # Build file names
        imgPath=os.path.join(PATH_DATASET_TRAIN_IMG,'IMG_%05d'%trainImageNumber+EXT_DATASET_IMAGE)
        lblPath=os.path.join(PATH_DATASET_TRAIN_LBL,'IMG_%05d'%trainImageNumber+EXT_DATASET_LABELS)
        pvwPath=os.path.join(PATH_DATASET_TRAIN_PVW,'IMG_%05d'%trainImageNumber+EXT_DATASET_IMAGE)
        trainImageNumber+=1
        # Get the image and the labels
        curImage,curLabels=dataSet.get_image()
        # Create the preview image
        curPreview=put_labels_in_image(curImage,curLabels,1)
        # Save images
        imsave(imgPath,curImage)
        imsave(pvwPath,curPreview)
        # Save labels
        if len(curLabels)==0:
            np.savetxt(lblPath,curLabels)
        else:
            np.savetxt(lblPath,curLabels,fmt=['%d']+['%.6f']*4)

    # Change the dataset to be the validation one
    dataSet.set_mosaic_ratio(mosaicYRatioMin=RATIO_TRAIN,mosaicYRatioMax=1)

    # Export the validation data items
    for idxImage in range(numValImagesPerMosaic):
        print('  + SAVING VALIDATION DATA %d of %d (MOSAIC %d of %d)'%(idxImage+1,numValImagesPerMosaic,idxMosaic+1,len(PATH_TRAIN_MOSAICS)))
        # Build the file names
        imgPath=os.path.join(PATH_DATASET_VALIDATION_IMG,'IMG_%05d'%validationImageNumber+EXT_DATASET_IMAGE)
        lblPath=os.path.join(PATH_DATASET_VALIDATION_LBL,'IMG_%05d'%validationImageNumber+EXT_DATASET_LABELS)
        pvwPath=os.path.join(PATH_DATASET_VALIDATION_PVW,'IMG_%05d'%validationImageNumber+EXT_DATASET_IMAGE)
        validationImageNumber+=1
        # Get the image and labels
        curImage,curLabels=dataSet.get_image()
        # Build the preview image
        curPreview=put_labels_in_image(curImage,curLabels,1)
        # Save the images
        imsave(imgPath,curImage)
        imsave(pvwPath,curPreview)
        # Save the labels
        if len(curLabels)==0:
            np.savetxt(lblPath,curLabels)
        else:
            np.savetxt(lblPath,curLabels,fmt=['%d']+['%.6f']*4)
    # Clean the dataset just in case.
    dataSet.clean()

# =============================================================================
# LOOP FOR EACH TEST MOSAIC
# =============================================================================

for idxMosaic,curMosaic in enumerate(PATH_TEST_MOSAICS):
    print('* PROCESSING MOSAIC %s (%d of %d)'%(curMosaic,idxMosaic+1,len(PATH_TEST_MOSAICS)))
    # Create the test dataset
    dataSet.create(mosaicFileName=curMosaic+EXT_MOSAIC_IMAGE,
                    labelsFileName=curMosaic+EXT_MOSAIC_LABELS,
                    imgWidth=WIDTH_IMAGE,
                    imgHeight=HEIGHT_IMAGE,
                    minArea=MIN_LABEL_AREA,
                    minLabels=MIN_LABELS_PER_IMAGE,
                    maxIter=MAX_SEARCH_ITERATIONS,
                    rejectPartial=REJECT_IMAGES_WITH_PARTIAL_LABELS)
    # Loop for each test data item
    for idxImage in range(numTestImagesPerMosaic):
        print('  + SAVING TEST DATA %d of %d (MOSAIC %d of %d)'%(idxImage+1,numTestImagesPerMosaic,idxMosaic+1,len(PATH_TEST_MOSAICS)))
        # Build the file names
        imgPath=os.path.join(PATH_DATASET_TEST_IMG,'IMG_%05d'%testImageNumber+EXT_DATASET_IMAGE)
        lblPath=os.path.join(PATH_DATASET_TEST_LBL,'IMG_%05d'%testImageNumber+EXT_DATASET_LABELS)
        pvwPath=os.path.join(PATH_DATASET_TEST_PVW,'IMG_%05d'%testImageNumber+EXT_DATASET_IMAGE)
        testImageNumber+=1
        # Get the image and labels
        curImage,curLabels=dataSet.get_image()
        # Create the preview image
        curPreview=put_labels_in_image(curImage,curLabels,1)
        # Save the images
        imsave(imgPath,curImage)
        imsave(pvwPath,curPreview)
        # Save the labels
        if len(curLabels)==0:
            np.savetxt(lblPath,curLabels)
        else:
            np.savetxt(lblPath,curLabels,fmt=['%d']+['%.6f']*4)
