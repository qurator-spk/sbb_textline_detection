# Textline-Recognition

***
# Introduction
This tool performs textline detection from image data and returns the results as PAGE-XML.

# Installation

`sudo pip install .`

# Models
In order to run this tool you also need trained models. You can download them here:   
https://file.spk-berlin.de:8443/textline_detection/

# Usage

`sbb_textline_detector -i <image file name> -o <directory to write output xml> -m <directory of models>`
