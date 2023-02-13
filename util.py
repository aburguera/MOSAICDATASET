#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : util
# Description : General purpose utility functions
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 27-Dec-2022 - Creation
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys,os

###############################################################################
# GENERAL PURPOSE FUNCTIONS
###############################################################################

# =============================================================================
# CREATE_DIRECTORY
# Attempts to create a directory. If it does not exist, creates it. If it
# exists and is empty, does nothing. It it exists and is not empty, raises an
# error.
# Input  : theFolder - Path of the directory to create.
#          createIfNotEmpty - If False and the destination path is not empty,
#                             an error will be raised.
# =============================================================================
def create_directory(theFolder,createIfNotEmpty=False):
    # If the folder does not exist, create it
    if not os.path.exists(theFolder):
        os.makedirs(theFolder,exist_ok=False)
        return
    # If the folder is not empty, abort
    if (not createIfNotEmpty) and len(os.listdir(theFolder))!=0:
        sys.exit('[ERROR] THE TARGET FOLDER %s IS NOT EMPTY.'%theFolder)

###############################################################################
# IMAGE AND LABELS TRANSFORMATION FUNCTIONS
###############################################################################

# =============================================================================
# ADD_VIGNETING
# Adds vigneting to an image.
# Input  : theImage - The image to which add vigneting. Must be in float
#                     format (values between 0 and 1). Can be color or gray-
#                     scale.
#          sigmaMultiplier - The base sigma is half the maximum image dimension
#                     The used sigma is sigmaMultiplier times that base value.
#          xRelCenter, yRelCenter - Central position of the vigneting expressed
#                     as values from 0 (left or top) to 1 (right or bottom).
# Output : The vigneted image.
# =============================================================================
def add_vigneting(theImage,sigmaMultiplier=1.0,xRelCenter=0.5,yRelCenter=0.5):
    # Compute sigma
    theSigma=sigmaMultiplier*max(theImage.shape[:2])/2.0
    # Compute central coordinates
    xCenter=int(theImage.shape[1]*xRelCenter)
    yCenter=int(theImage.shape[0]*yRelCenter)
    # Create a 2D grid of indices
    theY,theX=np.ogrid[:theImage.shape[0],:theImage.shape[1]]
    # Compute the distance from the center for each point in the grid
    theX-=xCenter
    theY-=yCenter
    # Compute the 2D Gaussian function
    theMask=np.exp(-(theX*theX+theY*theY)/(2*theSigma**2))
    # Normalize the mask so that all values are between 0 and 1
    theMask/=np.max(theMask)
    # Apply the mask to the image and return it
    if len(theImage.shape)==2:
        return theImage*theMask
    else:
        return theImage*np.repeat(theMask[:,:,np.newaxis],theImage.shape[2],axis=2)

# =========================================================================
# TRANSFORM_IMAGE
# Applies a random transform to the input image and labels.
# Input  : theImage - Image to transform.
#          theLabels - Labels to transform in ABSOLUTE format.
#          flipValue - Bit 0=1: Perform horizontal flip.
#                      Bit 1=1: Perform verticla flip.
#          brightValue - Value between 0 (darken image) and 1 (lighten va-
#                        lue). The maximum darken/lighten value is
#                        brightScale.
#          brightScale - Maximum brighten/darken value.
#          sigmaMultiplier - Multiplies the standard deviation used to
#                        simulate vigneting. See add_vigneting.
#          xRelCenter,yRelCenter - Vigneting effect center (specified as
#                        values from 0 to 1 relative to the image size).
# Output : theImage - Transformed image
#          theLabels - Transformed labels (ABSOLUTE format)
# =========================================================================
def transform_image(theImage,theLabels,flipValue,brightValue,brightScale,sigmaMultiplier,xRelCenter,yRelCenter):
    # If horizontal flip is necessary
    if (flipValue & 1)!=0:
        # Flip the image
        theImage=np.fliplr(theImage)
        # Flip the labels
        theLabels=[[curLabel[0],theImage.shape[1]-1-curLabel[3],curLabel[2],theImage.shape[1]-1-curLabel[1],curLabel[4],curLabel[5]] for curLabel in theLabels]
    # If vertical flip is necessary
    if (flipValue & 2)!=0:
        # Flip the image
        theImage=np.flipud(theImage)
        # Flip the labels
        theLabels=[[curLabel[0],curLabel[1],theImage.shape[0]-1-curLabel[4],curLabel[3],theImage.shape[0]-1-curLabel[2],curLabel[5]] for curLabel in theLabels]
    # Apply brightness change
    theImage=np.clip(theImage+(brightValue*brightScale*2)-brightScale,0,1)
    # Apply vigneting
    theImage=add_vigneting(theImage,sigmaMultiplier,xRelCenter,yRelCenter)
    # Return the image and the labels
    return theImage,theLabels

###############################################################################
# LABEL PROCESSING FUNCTIONS
###############################################################################

# =============================================================================
# CONVERT_OBJDEDECT_LABELS
# Convert a list of object detection labels from one format to another.
# The allowed formats for the labels in the list are:
# YOLOv5: [CLASS,XC,YC,W,H] where
#          CLASS: Integer. Object class.
#          XC,YC: Floats. Bounding box center expressed as values within the
#                 interval [0,1] relative the the tagged image width and height
#          W,H: Floats. Width and height of the bounding box expressed as
#               values within the interval [0,1] relative to the tagged image
#               width and height.
# ABSOLUTE: [CLASS,XL,YT,XR,YB,AREA] where:
#          CLASS: Integer. Object class.
#          XL,YT: Integers. Pixel coordinates of the top left corner of the
#                 bounding box.
#          XR,YB: Integers. Pixel coordinates of the bottom right corner of the
#                 bounding box.
#          AREA: Area of the bounding box, expressed in squared pixels. The
#                area is redundant, but it is useful to have it precomputed.
# Input  : theLabels - List of labels in the source format (YOLOv5 or ABSOLUTE)
#          imgWidth,imgHeight - Size of the tagged image, in pixels.
#          theConversion - 0 : Convert from YOLOv5 to ABSOLUTE
#                          1 : Convert from ABSOLUTE to YOLOv5
# Output : outLabels - List of labels in the destination format.
# =============================================================================
def convert_objdetect_labels(theLabels,imgWidth,imgHeight,theConversion):
    if theConversion==0:
        return [[int(curLabel[0]),
                 int(round((curLabel[1]-curLabel[3]/2)*(imgWidth-1))),
                 int(round((curLabel[2]-curLabel[4]/2)*(imgHeight-1))),
                 int(round((curLabel[1]+curLabel[3]/2)*(imgWidth-1))),
                 int(round((curLabel[2]+curLabel[4]/2)*(imgHeight-1))),
                 int(round(curLabel[3]*imgWidth*curLabel[4]*imgHeight))]
                for curLabel in theLabels]
    elif theConversion==1:
        return [[curLabel[0],
                 ((curLabel[1]+curLabel[3])/2)/(imgWidth-1),
                 ((curLabel[2]+curLabel[4])/2)/(imgHeight-1),
                 (curLabel[3]-curLabel[1])/(imgWidth-1),
                 (curLabel[4]-curLabel[2])/(imgHeight-1)]
                for curLabel in theLabels]
    else:
        sys.exit('[ERROR] WRONG LABEL CONVERSION SPEC %d. VALID VALUES ARE 0 AND 1'%theConversion)

# =========================================================================
# GET_LABELS_IN_RECTANGLE
# Given a list of labels and a rectangle, outputs the labels that sufficiently
# overlap with the rectangle.
# Input  : theLabels - Input labels in ABSOLUTE format.
#          xLeft,yTop,xRight,yBottom - Rectangle specs in pixels. The
#          rectangle goes from xLeft to xRight-1 and from yTop to yBottom-1.
#          minPct - The ratio between the overlapping area of a label and
#                   the total area must be larger or equal to minPct for
#                   the label to be accepted.
#          rejectPartial - If True, the search ends if a partially overlapping
#                   label is found.
# Output : outLabels - The labels in ABSOLUTE format and coordinates relative
#                      to the rectangle.
# =========================================================================
def get_labels_in_rectangle(theLabels,xLeft,yTop,xRight,yBottom,minPct=0.5,rejectPartial=False):
    outLabels=[]
    isPartial=False
    for curLabel in theLabels:
        # If the label is inside the image
        if curLabel[3]>=xLeft and curLabel[1]<xRight and curLabel[4]>=yTop and curLabel[2]<yBottom:
            # Compute the coordinates
            lxLeft=max(0,curLabel[1]-xLeft)
            lyTop=max(0,curLabel[2]-yTop)
            lxRight=min(xRight-xLeft,curLabel[3]-xLeft)
            lyBottom=min(yBottom-yTop,curLabel[4]-yTop)
            # Consider the label only if the box area is large enough
            theArea=((lxRight-lxLeft)*(lyBottom-lyTop))
            areaRatio=theArea/curLabel[5]
            # If the label is partial and rejectPartial is True, exit
            if rejectPartial and areaRatio<1:
                isPartial=True
                break
            if areaRatio>=minPct:
                outLabels.append([curLabel[0],lxLeft,lyTop,lxRight,lyBottom,theArea])
    return outLabels,isPartial

###############################################################################
# LABEL AND IMAGE VISUALIZATION FUNCTIONS
###############################################################################

# =============================================================================
# PLOT_IMAGE_WITH_YOLO_LABELS
# Plots an image with the specified labels (YOLOv5 format) overlayed.
# Input  : theImage - The image to show.
#          theLabels - List of YOLOv5 formatted annotations
#          showClass - Show the class ID (ugly)
#          boxColor - matplotlib color spec to draw the labels boxes.
# =============================================================================
def plot_image_with_yolo_labels(theImage,theLabels,showClass=False,boxColor='w'):
    plt.figure()
    theAxes=plt.gca()
    if len(theImage.shape)==2 or theImage.shape[2]==1:
        plt.imshow(theImage,cmap='gray')
    else:
        plt.imshow(theImage)
    for curLabel in theLabels:
        curRect=patches.Rectangle((((curLabel[1]-(curLabel[3]/2))*(theImage.shape[1]-1)),((curLabel[2]-(curLabel[4]/2))*(theImage.shape[0]-1))),curLabel[3]*(theImage.shape[1]-1),curLabel[4]*(theImage.shape[0]-1),edgecolor=boxColor,fill=False)
        theAxes.add_patch(curRect)
        if showClass:
            theAxes.annotate('%d'%curLabel[0],(curLabel[1]*(theImage.shape[1]-1),curLabel[2]*(theImage.shape[0]-1)),ha='center',va='center',color=boxColor)
    plt.show()

# =============================================================================
# PUT_LABELS_IN_IMAGE
# Given an image and a set of object detection labels, modifies the image so
# that the labels are shown. Note that this *modifies* the image. Its main
# purpose is to save a preview of the tagged image for human visual inspection.
# Input  : theImage - Image to modify.
#          thelabels - Labels to plot.
#          lblFormat - Label format (0: ABSOLUTE, 1: YOLOv5). For more infor-
#                      mation on the formats, check CONVERT_OBJDEDECT_LABELS.
# Output : pvwImage - The modified image
# =============================================================================
def put_labels_in_image(theImage,theLabels,lblFormat):
    # Copy the image to avoid modifying the input one.
    pvwImage=theImage.copy()
    # Convert labels if necessary
    if lblFormat==1:
        pvwLabels=convert_objdetect_labels(theLabels,theImage.shape[1],theImage.shape[0],0)
    else:
        pvwLabels=theLabels.copy()
    # Build the "white" color depending on the image format
    if np.issubdtype(theImage.dtype,np.integer):
        vWhite=255
    else:
        vWhite=1
    if len(theImage.shape)==3 and theImage.shape[2]==3:
        vWhite=[vWhite,vWhite,vWhite]
    # Draw every label
    for curLabel in pvwLabels:
        # Compute the rectangle coordinates
        [theRows,theColumns]=rectangle_perimeter((curLabel[2],curLabel[1]),(curLabel[4],curLabel[3]),shape=theImage.shape[:2],clip=True)
        # Modify the image
        pvwImage[theRows,theColumns]=vWhite
    # Return the modified image
    return pvwImage