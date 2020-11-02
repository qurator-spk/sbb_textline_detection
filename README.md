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
For the purpose of text recognition (OCR) and in order to avoid noise being introduced from texts outside the printspace, one first needs to detect the border of the printed frame. This is done by a binary pixelwise-segmentation model trained on a dataset of 2,000 documents where about 1,200 of them come from the [dhSegment](https://github.com/dhlab-epfl/dhSegment/) project (you can download the dataset from [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip)) and the remainder having been annotated in SBB. For border detection, the model needs to be fed with the whole image at once rather than separated in patches.

## Layout detection
As a next step, text regions need to be identified by means of layout detection. Again a pixelwise segmentation model was trained on 131 labeled images from the SBB digital collections, including some data augmentation. Since the target of this tool are historical documents, we consider as main region types text regions, separators, images, tables and background - each with their own subclasses, e.g. in the case of text regions, subclasses like header/heading, drop capital, main body text etc. While it would be desirable to detect and classify each of these classes in a granular way, there are also limitations due to having a suitably large and balanced training set. Accordingly, the current version of this tool is focussed on the main region types background, text region, image and separator. 

## Textline detection
In a subsequent step, binary pixelwise segmentation is used again to classify pixels in a document that constitute textlines. For textline segmentation, a model was initially trained on documents with only one column/block of text and some augmentation with regards to scaling. By fine-tuning the parameters also for multi-column documents, additional training data was produced that resulted in a much more robust textline detection model.

## Heuristic methods
Some heuristic methods are also employed to further improve the model predictions: 
* After border detection, the largest contour is determined by a bounding box and the image cropped to these coordinates. 
* For text region detection, the image is scaled up to make it easier for the model to detect background space between text regions.
* A minimum area is defined for text regions in relation to the overall image dimensions, so that very small regions that are actually noise can be filtered out. 
* Deskewing is applied on the text region level (due to regions having different degrees of skew) in order to improve the textline segmentation result. 
* After deskewing, a calculation of the pixel distribution on the X-axis allows the separation of textlines (foreground) and background pixels.
* Finally, using the derived coordinates, bounding boxes are determined for each textline.

## Installation
`pip install .`

### Models
In order to run this tool you also need trained models. You can download our pretrained models from here:   
https://qurator-data.de/sbb_textline_detector/

## Usage

The basic command-line interface can be called like this:

    sbb_textline_detector -i <image file name> -o <directory to write output xml> -m <directory of models>

The tool does accept raw (RGB/grayscale) images as input, but results will be much improved when a properly binarized image is used instead. We also provide a [tool](https://github.com/qurator-spk/sbb_binarization) to perform this binarization step.

### Usage with OCR-D

In addition, there is a CLI for [OCR-D](https://ocr-d.de/en/spec/cli):

    ocrd-sbb-textline-detector -I OCR-D-IMG -O OCR-D-SEG-LINE-SBB -P model /path/to/the/models/textline_detection

Segmentation works on raw (RGB/grayscale) images, but honours `AlternativeImage`s from earlier preprocessing steps, so it's OK to perform (say) deskewing first, followed by textline detection. Results from previous cropping or binarization steps are allowed and retained, but will be ignored. (So these are only useful if themselves needed for deskewing or dewarping prior to segmentation.) 

This processor will replace any previously existing `Border`, `ReadingOrder` and `TextRegion` instances (but keep other region types unchanged).
