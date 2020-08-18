[![Build Status](https://travis-ci.org/qurator-spk/sbb_textline_detection.svg?branch=master)](https://travis-ci.org/qurator-spk/sbb_textline_detection)

# Textline Detection
> Detect textlines in document images

## Introduction
This tool performs border, region and textline detection from document image data and returns the results as [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML).
The goal of this project is to extract textlines of a document in order to feed them to an OCR model. This is achieved by four successive stages as follows:
* [Border detection](https://github.com/qurator-spk/sbb_textline_detection#border-detection)
* [Layout detection](https://github.com/qurator-spk/sbb_textline_detection#layout-detection)
* [Textline detection](https://github.com/qurator-spk/sbb_textline_detection#textline-detection)
* [Heuristic methods](https://github.com/qurator-spk/sbb_textline_detection#heuristic-methods)

The first three stages are based on [pixelwise segmentation](https://github.com/qurator-spk/sbb_pixelwise_segmentation).

## Border detection
For the purpose of text recognition (OCR) and in order to avoid noise being introduced from texts outside the printspace, one first needs to detect the border of the printed frame. This is done by a binary pixelwise-segmentation model trained on a dataset of 2,000 documents where about 1,200 of them come from the [dhSegment](https://github.com/dhlab-epfl/dhSegment/) project (you can download the dataset from [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip) and the remainder having been annotated in SBB. For border detection, the model needs to be fed with the whole image at once rather than separated in patches.

## Layout detection
As a next step text regions need to be identified in layout detection. Again a pixelwise segmentation is implemented. For this purpose we had to provide training images and labels. In SBB we have a good resources and gt for layout of documents. By historical documents we have some main regions like , textregion, separators, images, tables and background and each has its own subclasses. For example for textregions we have subclasses like header, heading, drop capitals , main text and etc. As we can see we have many classes and in fact the ideal is to classify documents based on all those classes. But this is really a tough job and since here our focus is on ocr we decided to train our model with main regions including background, textregions, images and separators. <br/>
We have used 131 documents to train our model. Of course augmentation also hase done but here I do not want to explain training process in detail.

## Textline detection
Last step is to do a binary pixelwise segmentation in order to classify textline pixels in document. For textline segmentation we had GT of documents with only one columns. This means that scale of documents were almost same , we tried to resolve this by feeding model with different scales of documents. However, even with this augmentation it was not easy to cover all spectrum of scales. So, this time we tried to use trained model and with tuning the parameters for multicolumns documents detect textlines. We then used this results also as GT to train new model which was much more robust. 

## Heuristic methods
After training models, we have used them to predict and classify documents in each step and then tried to use results to extract desirable textlines recatngles.<br/>
After applying page extraction model we then found the biggest contour and after fitting a bounding box we were able to crop image inside this box.<br/>
By layout detection, it was so important for us to detect textregions clearly separately that is why we have scaled image up. With this trick it was easier for model to detect background spaces between textregions. <br/>
We have set a minimum textregion area in respect to area of whole image, so those small textregions which are actullay noises in our prediction are filtered. At the end we have found contours of textregions  and corresponding boundin boxes. <br/>
Textline segmentation is also done and using bounding boxes from textregions we are now able to get textline segmentation for each individual textregion. The first thing that we face by historical documents is that documents are skewed and even worser that each textregion can be skewed in a differnt manner. So, it was a key feature to deskew each textregion. Actually we have used textline segmengtation in each region to deskew corresponding region. After deskewing , calculating distribution of textlines segmentation result in X-direction  helped us to find starting and ending point of every single textline. You can imagine that the hills in mentioned distribution are actully where we have background and no textline. Finally, using this coordinates we were able to find bounding rectangle for each textline.



## Installation
`pip install .`

### Models
In order to run this tool you also need trained models. You can download our pretrained models from here:   
https://qurator-data.de/sbb_textline_detector/

## Usage
`sbb_textline_detector -i <image file name> -o <directory to write output xml> -m <directory of models>`

### Usage with OCR-D
~~~
ocrd-example-binarize -I OCR-D-IMG -O OCR-D-IMG-BIN
ocrd-sbb-textline-detector -I OCR-D-IMG-BIN -O OCR-D-SEG-LINE-SBB \
        -p '{ "model": "/path/to/the/models/textline_detection" }'
~~~

Segmentation works on raw RGB images, but retains
`AlternativeImage`s from binarization steps, so it's OK to do
binarization first, then perform the textline detection. The used binarization
processor must produce an `AlternativeImage` for the binarized image, not
replace the original raw RGB image.
