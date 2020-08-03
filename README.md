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
<br/>
First three stages are done by using a pixel-wise segmentation. You can train your own model using this tool (https://github.com/qurator-spk/sbb_pixelwise_segmentation).

## Printspace or border extraction
From ocr point of view and in order to avoid texts outside printspace region, you need to detect and extract printspace region. As mentioned briefly earlier this is done by a binary pixelwise-segmentation. We have trained our model by a dataset of 2000 documents where about 1200 of them was from dhsegment project (you can download the dataset from here https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip) and the rest was annotated by myself using our dataset in SBB. 
This is worthy to mention that for page (printspace or border) extractation you have to feed model whole image at once and not in patches.


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
