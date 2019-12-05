# Textline-Recognition

## Introduction
This tool performs textline detection from document image data and returns the results as PAGE-XML.

## Installation

`pip install .`

## Models
In order to run this tool you also need trained models. You can download our pre-trained models from here:   
https://file.spk-berlin.de:8443/textline_detection/

## Usage

`sbb_textline_detector -i <image file name> -o <directory to write output xml> -m <directory of models>`
