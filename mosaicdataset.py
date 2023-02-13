# -*- coding: utf-8 -*-

##############################################################################
# Name        : MosaicDataSet
#
# Description : Creates random labeled images from a large labeled image.
#
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 24-Jan-2023 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from skimage.io import imread
from skimage import img_as_float,img_as_ubyte
from util import convert_objdetect_labels,get_labels_in_rectangle,plot_image_with_yolo_labels,transform_image

###############################################################################
# MOSAICDATASET CLASS
###############################################################################

class MosaicDataSet:

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================
    def __init__(self):
        self.clean()

    # =========================================================================
    # CLEAN
    # Sets all members to None
    # =========================================================================
    def clean(self):
        self._mosaicFileName=None
        self._labelsFileName=None
        self._imgWidth=None
        self._imgHeight=None
        self._mosaicXRatioMin=None
        self._mosaicXRatioMax=None
        self._mosaicYRatioMin=None
        self._mosaicYRatiomax=None
        self._minLabels=None
        self._minArea=None
        self._maxIter=None
        self._brightScale=None
        self._minVSigma=None
        self._maxVSigma=None
        self._minVX=None
        self._maxVX=None
        self._minVY=None
        self._maxVY=None
        self._theMosaic=None
        self._theLabels=None
        self._mosaicWidth=None
        self._mosaicHeight=None
        self._mosaicXMin=None
        self._mosaicXMax=None
        self._rejectPartial=None

    # =========================================================================
    # CREATE
    # Sets main dataset parameters and load files.
    # Input : mosaicFileName - Full path to the mosaic image file.
    #         labelsFileName - Full path to the YOLO-formatted mosaic labels.
    #         imgWidth,imgHeight - Size of the output images.
    #         mosaicXRatioMin,... Defines a box of the part of the mosaic
    #                   that can be used to generate images. Useful to
    #                   split a mosaic in train, validation and test. Values
    #                   are between 0 and 1 (0 top or left, 1 bottom or right)
    #         minLabels - Minimum number of labels per output image.
    #         minArea - Minimum area of a label image relative to the total
    #                   label area.
    #         maxIter - Maximum number of iterations to find the image with
    #                   the requested number of labels (minLabels) and
    #                   area (minArea). If not found, the best one is returned.
    #         brightScale - Maximum change when darkening and lightning (data
    #                   augmentation).
    #         minVS,maxVX,minVY,maxVY - Range of the vigneting light focus,
    #                   specified as values between 0 (left, up) to 1 (right,
    #                   bottom). Data augmentation.
    #         minVSigma,maxVSigma - Range of values for the standar deviation
    #                   of the Gaussian used to simulate vigneting. Also
    #                   data augmentation.
    #         rejectPatial - True if images with partial labels are not
    #                   considered.
    #         randomSeed - Used to initialize the random seed. Set to a speci-
    #                   value to ensure the same images are generated or
    #                   use None to use the standard initialization.
    # =========================================================================
    def create(self,mosaicFileName,labelsFileName,imgWidth=640,imgHeight=480,mosaicXRatioMin=0,mosaicXRatioMax=1,mosaicYRatioMin=0,mosaicYRatioMax=1,minLabels=1,minArea=0.5,maxIter=1000,brightScale=0.1,minVX=0.4,maxVX=0.6,minVY=0.4,maxVY=0.6,minVSigma=0.5,maxVSigma=1.25,rejectPartial=False,randomSeed=None):
        # Initialize random seed
        np.random.seed(randomSeed)
        # Store parameters
        self._mosaicFileName=mosaicFileName
        self._labelsFileName=labelsFileName
        self._imgWidth=imgWidth
        self._imgHeight=imgHeight
        self._minLabels=minLabels
        self._minArea=minArea
        self._maxIter=maxIter
        self._brightScale=brightScale
        self._minVSigma=minVSigma
        self._maxVSigma=maxVSigma
        self._minVX=minVX
        self._maxVX=maxVX
        self._minVY=minVY
        self._maxVY=maxVY
        self._rejectPartial=rejectPartial
        # Load the mosaic
        self._theMosaic=img_as_float(imread(mosaicFileName))
        # Load the YOLO formatted labels and convert them to ABSOLUTE format.
        self._theLabels=convert_objdetect_labels(np.loadtxt(labelsFileName,delimiter=' '),self._theMosaic.shape[1],self._theMosaic.shape[0],0)
        # Store some related parameters
        self._mosaicWidth=self._theMosaic.shape[1]
        self._mosaicHeight=self._theMosaic.shape[0]
        # Set the mosaic ratio (useable region)
        self.set_mosaic_ratio(mosaicXRatioMin, mosaicXRatioMax, mosaicYRatioMin, mosaicYRatioMax)

    # =========================================================================
    # SET_MOSAIC_RATIO
    # Changes the part of the mosaic used to generate images. Useful to
    # separate one single mosaic in train and validation, ...
    # Input  : See parameters of the same name in CREATE.
    # =========================================================================
    def set_mosaic_ratio(self,mosaicXRatioMin=0,mosaicXRatioMax=1,mosaicYRatioMin=0,mosaicYRatioMax=1):
        self._mosaicXRatioMin=mosaicXRatioMin
        self._mosaicXRatioMax=mosaicXRatioMax
        self._mosaicYRatioMin=mosaicYRatioMin
        self._mosaicYRatioMax=mosaicYRatioMax
        self._mosaicXMin=self._mosaicXRatioMin*self._mosaicWidth
        self._mosaicXMax=self._mosaicXRatioMax*self._mosaicWidth
        self._mosaicYMin=self._mosaicYRatioMin*self._mosaicHeight
        self._mosaicYMax=self._mosaicYRatioMax*self._mosaicHeight

    # =========================================================================
    # GET_IMAGE
    # Provides a random labeled image taken from the mosaic and modified
    # randomly using transform_image.
    # Input  : outUBYTE - True if the image must have UBYTE format or False
    #                     for a float format.
    #          lblFormat - Labels format. 0 ABSOLUTE, 1 YOLO
    # Output : outImage - The image.
    #          bestLabels - The labels
    # =========================================================================
    def get_image(self,outUBYTE=True,lblFormat=1):
        maxLabels=0
        for i in range(self._maxIter):
            doSearch=True
            # This loop ensures that no images with partial labels are consi-
            # dered if _rejectPartial is True. Note that this can lead to
            # an infinite loop (though quite unlikely).
            while doSearch:
                # Generate random rectangle of the desired image size
                xLeft=np.random.randint(self._mosaicXMin,self._mosaicXMax-self._imgWidth)
                yTop=np.random.randint(self._mosaicYMin,self._mosaicYMax-self._imgHeight)
                xRight=xLeft+self._imgWidth
                yBottom=yTop+self._imgHeight
                # Search labels
                curLabels,isPartial=get_labels_in_rectangle(self._theLabels,xLeft,yTop,xRight,yBottom,self._minArea,self._rejectPartial)
                # Repeat only if partial labels and _rejectPartial is True
                doSearch=self._rejectPartial and isPartial
            numLabels=len(curLabels)
            # Store the data if the current rectangle has more labels
            if numLabels>maxLabels or maxLabels==0:
                maxLabels=numLabels
                bestRect=[xLeft,yTop,xRight,yBottom]
                bestLabels=curLabels
            # If the minimum number of labels is reached, exit
            if numLabels>=self._minLabels:
                break
        # Get the image
        outImage=self._theMosaic[bestRect[1]:bestRect[3],bestRect[0]:bestRect[2]]
        # Apply random transform
        outImage,bestLabels=transform_image(outImage,bestLabels,np.random.randint(4),np.random.random(),self._brightScale,np.random.random()*(self._maxVSigma-self._minVSigma)+self._minVSigma,np.random.random()*(self._maxVX-self._minVX)+self._minVX,np.random.random()*(self._maxVY-self._minVY)+self._minVY)
        # Convert the labels format
        bestLabels=convert_objdetect_labels(bestLabels,self._imgWidth,self._imgHeight,lblFormat)
        # Convert to unsigned byte format if requested
        if outUBYTE:
            outImage=img_as_ubyte(outImage)
        # Output the data
        return outImage,bestLabels

    # =========================================================================
    # PLOT
    # Plots the labeled mosaic.
    # Input : showClass - Show the class number (True) or not (False).
    #         boxColor - The color to plot the bounding boxes.
    # =========================================================================
    def plot(self,showClass=False,boxColor='w'):
        plot_image_with_yolo_labels(self._theMosaic,convert_objdetect_labels(self._theLabels,self._mosaicWidth,self._mosaicHeight,1),showClass=showClass, boxColor=boxColor)