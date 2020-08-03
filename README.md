# Textline Detection
> Detect textlines in document images

## Introduction
This tool performs printspace, region and textline detection from document image
data and returns the results as [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML).
The goal of this project is to extract textlines of a document to feed an ocr model. This is achieved by four successive stages as follows:
* Printspace or border extraction
* Layout analysis
* Textline detection
* Heuristic methods

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
