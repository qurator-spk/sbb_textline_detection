# Textline Detection

## Introduction
This tool performs textline detection from document image data and returns the results as PAGE-XML.

## Installation

`pip install .`

## Models
In order to run this tool you also need trained models. You can download our pre-trained models from here:   
https://file.spk-berlin.de:8443/textline_detection/

## Usage

`sbb_textline_detector -i <image file name> -o <directory to write output xml> -m <directory of models>`

## Usage with OCR-D

~~~
ocrd-example-binarize -I OCR-D-IMG -O OCR-D-IMG-BIN
ocrd-sbb-textline-detector -I OCR-D-IMG-BIN -O OCR-D-SEG-LINE-SBB \
        -p '{ "model": "/path/to/the/models/textline_detection" }'
~~~

Segmentation works on raw RGB images, but respects and retains
`AlternativeImage`s from binarization steps, so it's a good idea to do
binarization first, then perform the textline detection. The used binarization
processor must produce an `AlternativeImage` for the binarized image, not
replace the original raw RGB image.
