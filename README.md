# Object Detection Dataset Creation from Labeled Mosaics

This software facilitates building Object Detection datasets (YOLOv5 format) by picking a large, labeled, mosaic and creating the requested number of images and labels from that mosaic. The images are randomly picked from the input mosaic and the corresponding labels adapted to each image local coordinate system.

Data augmentation is applied to the images in the form of random flipping, brightness, contrast and vigneting.

It is also possible to specify the minumum percentage of the source label area that must lie inside an image to consider the label and the minimum number of labels in an image to consider the image.

## Credits and citation

**Science and code**: Antoni Burguera (antoni dot burguera at uib dot es)

**Citation**: If you use this software, please cite the following paper:

Reference to be posted soon. Please contact us.

## Basic usage

Check **create_dataset.py** to see a usage example. The mosaics and labels used in this example are not available in this repository.

All files are fully commented. Check the comments in each file to learn about it.

## Understanding the system

The main class is **MosaicDataSet**. To use it, just instantiate an object and then specify the desired parameters with the **create** method. The parameters, which are fully commented in the code, allow stating the input mosaic and label files, the part of the mosaic to be used, the desired data augmentation parameters and the criteria to accept labels and images.

Afterwards, every call to **get_image** will provide one new labeled image.

Also, note that the helper function **put_labels_in_image** creates an image with the labels plotted into it. This helps in creating a preview of the image and the bounding boxes.

Here is a basic example:

```
# Import packages
from mosaicdataset import MosaicDataSet
import matplotlib.pyplot as plt

# Create the dataset creator with default parameters.
dataSet=MosaicDataSet()
dataSet.create('MOSAIC_IMAGE.png','MOSAIC_YOLOV5LABELS.txt')

# Get one image and the corresponding labels.
theImage,theLabels=dataSet.get_image()

# Plot the labels into the image to ease visualization and testing.
previewImage=put_labels_in_image(theImage,theLabels,1)

# Display it
plt.figure()
plt.imshow(previewImage)
plt.show()
```

Aside of **put_labels_in_image**, some additional helper function are available in **util.py**.

## Requirements

To execute this software, you will need:

* Python 3
* NumPy
* Matplotlib
* SciKit-Image

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it or, eventually, create a time paradox, the result of which could cause a chain reaction that would unravel the very fabric of the space-time continuum and destroy the entire universe.