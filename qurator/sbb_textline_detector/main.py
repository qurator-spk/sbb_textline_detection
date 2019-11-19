#! /usr/bin/env python3

__version__ = '1.0'

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
import random
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model
import math
from shapely import geometry
from sklearn.cluster import KMeans
import gc
from keras import backend as K
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
import warnings
import click
import time
from multiprocessing import Process, Queue, cpu_count
import datetime


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

__doc__ = \
    """
    tool to extract table form data from alto xml data
    """


class textlineerkenner:
    def __init__(self, image_dir, dir_out, f_name, dir_models):
        self.image_dir = image_dir  # XXX This does not seem to be a directory as the name suggests, but a file
        self.dir_out = dir_out
        self.f_name = f_name
        if self.f_name is None:
            try:
                self.f_name = image_dir.split('/')[len(image_dir.split('/')) - 1]
                self.f_name = self.f_name.split('.')[0]
            except:
                self.f_name = self.f_name.split('.')[0]
        self.dir_models = dir_models
        self.kernel = np.ones((5, 5), np.uint8)
        self.model_page_dir = dir_models + '/model_page.h5'
        self.model_region_dir = dir_models + '/model_strukturerkennung.h5'
        self.model_textline_dir = dir_models + '/model_textline.h5'

    def find_polygons_size_filter(self, contours, median_area, scaler_up=1.2, scaler_down=0.8):
        found_polygons_early = list()

        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            # Check that polygon has area greater than minimal area
            if area >= median_area * scaler_down and area <= median_area * scaler_up:
                found_polygons_early.append(
                    np.array([point for point in polygon.exterior.coords], dtype=np.uint))
        return found_polygons_early

    def filter_contours_area_of_image(self, image, contours, hirarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(
                    image.shape[:2]) and hirarchy[0][jv][3] == -1 :  # and hirarchy[0][jv][3]==-1 :
                found_polygons_early.append(
                    np.array([ [point] for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1
        return found_polygons_early

    def filter_contours_area_of_image_interiors(self, image, contours, hirarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and \
                    hirarchy[0][jv][3] != -1:
                # print(c[0][0][1])
                found_polygons_early.append(
                    np.array([point for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1
        return found_polygons_early

    def resize_image(self, img_in, input_height, input_width):
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

    def resize_ann(self, seg_in, input_height, input_width):
        return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

    def get_one_hot(self, seg, input_height, input_width, n_classes):
        seg = seg[:, :, 0]
        seg_f = np.zeros((input_height, input_width, n_classes))
        for j in range(n_classes):
            seg_f[:, :, j] = (seg == j).astype(int)
        return seg_f


    def color_images(self, seg, n_classes):
        ann_u = range(n_classes)
        if len(np.shape(seg)) == 3:
            seg = seg[:, :, 0]

        seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(np.uint8)
        colors = sns.color_palette("hls", n_classes)

        for c in ann_u:
            c = int(c)
            segl = (seg == c)
            seg_img[:, :, 0] = segl * c
            seg_img[:, :, 1] = segl * c
            seg_img[:, :, 2] = segl * c
        return seg_img

    def color_images_diva(self, seg, n_classes):
        ann_u = range(n_classes)
        if len(np.shape(seg)) == 3:
            seg = seg[:, :, 0]

        seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(float)
        # colors=sns.color_palette("hls", n_classes)
        colors = [[1, 0, 0], [8, 0, 0], [2, 0, 0], [4, 0, 0]]

        for c in ann_u:
            c = int(c)
            segl = (seg == c)
            seg_img[:, :, 0][seg == c] = colors[c][0]  # segl*(colors[c][0])
            seg_img[:, :, 1][seg == c] = colors[c][1]  # seg_img[:,:,1]=segl*(colors[c][1])
            seg_img[:, :, 2][seg == c] = colors[c][2]  # seg_img[:,:,2]=segl*(colors[c][2])
        return seg_img

    def rotate_image(self, img_patch, slope):
        (h, w) = img_patch.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, slope, 1.0)
        return cv2.warpAffine(img_patch, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def cleaning_probs(self, probs: np.ndarray, sigma: float) -> np.ndarray:
        # Smooth
        if sigma > 0.:
            return cv2.GaussianBlur(probs, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
        elif sigma == 0.:
            return cv2.fastNlMeansDenoising((probs * 255).astype(np.uint8), h=20) / 255
        else:  # Negative sigma, do not do anything
            return probs

    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

    def otsu_copy(self, img):
        img_r = np.zeros(img.shape)
        img1 = img[:, :, 0]
        img2 = img[:, :, 1]
        img3 = img[:, :, 2]
        # print(img.min())
        # print(img[:,:,0].min())
        # blur = cv2.GaussianBlur(img,(5,5))
        # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retval2, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retval3, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_r[:, :, 0] = threshold1
        img_r[:, :, 1] = threshold1
        img_r[:, :, 2] = threshold1
        return img_r

    def get_image_and_scales(self):
        self.image = cv2.imread(self.image_dir)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]

        if self.image.shape[0] < 1000:
            self.img_hight_int = 2800
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 2000 and self.image.shape[0] >= 1000:
            self.img_hight_int = 3500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 3000 and self.image.shape[0] >= 2000:
            self.img_hight_int = 5500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 4000 and self.image.shape[0] >= 3000:
            self.img_hight_int = 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        else:
            self.img_hight_int = self.image.shape[0]
            self.img_width_int = self.image.shape[1]

        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = self.resize_image(self.image, self.img_hight_int, self.img_width_int)

    def start_new_session_and_model(self, model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session
    
    def do_prediction(self,patches,img,model):
        
        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]

        if patches:

            margin = int(0.1 * img_width_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                    

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
                
        if not patches:

            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, self.image.shape[0], self.image.shape[1])
            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true
            
        

    def extract_page(self):
        patches=False
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        img = self.otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(img, (15, 15), 0)

        
        img_page_prediction=self.do_prediction(patches,img,model_page)
        
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)

        box = [x, y, w, h]

        croped_page, page_coord = self.crop_image_inside_box(box, self.image)
        
        self.cont_page=[]
        self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] , 
                                                    [ page_coord[3] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[1] ] ,
                                                [ page_coord[2] , page_coord[1] ]] ) )

        session_page.close()
        del model_page
        del session_page
        del self.image
        del contours
        del thresh
        del img

        gc.collect()
        return croped_page, page_coord

    def extract_text_regions(self, img):
        
        patches=True
        model_region, session_region = self.start_new_session_and_model(self.model_region_dir)
        img = self.otsu_copy(img)
        img = img.astype(np.uint8)
        
        prediction_regions=self.do_prediction(patches,img,model_region)
        
        session_region.close()
        del model_region
        del session_region
        gc.collect()
        return prediction_regions

    def get_text_region_contours_and_boxes(self, image):
        rgb_class_of_texts = (1, 1, 1)
        mask_texts = np.all(image == rgb_class_of_texts, axis=-1)

        image = np.repeat(mask_texts[:, :, np.newaxis], 3, axis=2) * 255
        image = image.astype(np.uint8)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)


        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contours = self.filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.00001)
        self.boxes = []
        
        for jj in range(len(main_contours)):
            x, y, w, h = cv2.boundingRect(main_contours[jj])
            self.boxes.append([x, y, w, h])
            

        return main_contours

    def get_all_image_patches_coordination(self, image_page):
        self.all_box_coord=[]
        for jk in range(len(self.boxes)):
            _,crop_coor=self.crop_image_inside_box(self.boxes[jk],image_page)
            self.all_box_coord.append(crop_coor) 
        

    def textline_contours(self, img):
        patches=True
        model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir)
        img = self.otsu_copy(img)
        img = img.astype(np.uint8)
        
        prediction_textline=self.do_prediction(patches,img,model_textline)

        session_textline.close()

        del model_textline
        del session_textline
        gc.collect()
        return prediction_textline[:,:,0]

    def get_textlines_for_each_textregions(self, textline_mask_tot, boxes):
        textline_mask_tot = cv2.erode(textline_mask_tot, self.kernel, iterations=1)
        self.area_of_cropped = []
        self.all_text_region_raw = []
        for jk in range(len(boxes)):
            crop_img, crop_coor = self.crop_image_inside_box(boxes[jk],
                                                             np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img=crop_img.astype(np.uint8)
            self.all_text_region_raw.append(crop_img[:, :, 0])
            self.area_of_cropped.append(crop_img.shape[0] * crop_img.shape[1])

    def seperate_lines(self, img_path, contour_text_interest, thetha):
        (h, w) = img_path.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180. * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
        contour_text_interest_copy = contour_text_interest.copy()

        x_cont = contour_text_interest[:, 0, 0]
        y_cont = contour_text_interest[:, 0, 1]
        x_cont = x_cont - np.min(x_cont)
        y_cont = y_cont - np.min(y_cont)

        x_min_cont = 0
        x_max_cont = img_path.shape[1]
        y_min_cont = 0
        y_max_cont = img_path.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        mada_n = img_path.sum(axis=1)

        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = mada_n[:]  # [first_nonzero:last_nonzero]
        y_help = np.zeros(len(y) + 40)
        y_help[20:len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real)<=2 and len(peaks_real)>1:
            sigma_gaus=10
        else:
            sigma_gaus=8
    
    
        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)
        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg = peaks_neg - 20 - 20
        peaks = peaks - 20

        for jj in range(len(peaks_neg)):
            if peaks_neg[jj] > len(x) - 1:
                peaks_neg[jj] = len(x) - 1

        for jj in range(len(peaks)):
            if peaks[jj] > len(x) - 1:
                peaks[jj] = len(x) - 1

        textline_boxes = []
        textline_boxes_rot = []

        if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
            for jj in range(len(peaks)):
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(
                    1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)

                if point_down >= img_path.shape[0]:
                    point_down = img_path.shape[0] - 2

                if point_down_narrow >= img_path.shape[0]:
                    point_down_narrow = img_path.shape[0] - 2

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                    x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0

                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))

        elif len(peaks) < 1:
            pass

        elif len(peaks) == 1:
            x_min = x_min_cont
            x_max = x_max_cont

            y_min = y_min_cont
            y_max = y_max_cont

            p1 = np.dot(rotation_matrix, [int(x_min), int(y_min)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(y_min)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(y_max)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(y_max)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
            
            
            if x_min_rot1<0:
                x_min_rot1=0
            if x_min_rot4<0:
                x_min_rot4=0
            if point_up_rot1<0:
                point_up_rot1=0
            if point_up_rot2<0:
                point_up_rot2=0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(y_min)],
                                            [int(x_max), int(y_min)],
                                            [int(x_max), int(y_max)],
                                            [int(x_min), int(y_max)]]))



        elif len(peaks) == 2:
            dis_to_next = np.abs(peaks[1] - peaks[0])
            for jj in range(len(peaks)):
                if jj == 0:
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                    if point_up < 0:
                        point_up = 1
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
                elif jj == 1:
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
                    if point_down >= img_path.shape[0]:
                        point_down = img_path.shape[0] - 2
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                    x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0

                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))
        else:
            for jj in range(len(peaks)):

                if jj == 0:
                    dis_to_next = peaks[jj + 1] - peaks[jj]
                    # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                    if point_up < 0:
                        point_up = 1
                    # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
                elif jj == len(peaks) - 1:
                    dis_to_next = peaks[jj] - peaks[jj - 1]
                    # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
                    if point_down >= img_path.shape[0]:
                        point_down = img_path.shape[0] - 2
                    # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                else:
                    dis_to_next_down = peaks[jj + 1] - peaks[jj]
                    dis_to_next_up = peaks[jj] - peaks[jj - 1]

                    point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next_up)
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next_down)

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                    x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

                p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
                p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
                p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
                p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

                x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
                x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
                x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
                x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
                
                
                if x_min_rot1<0:
                    x_min_rot1=0
                if x_min_rot4<0:
                    x_min_rot4=0
                if point_up_rot1<0:
                    point_up_rot1=0
                if point_up_rot2<0:
                    point_up_rot2=0
                    


                textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                    [int(x_max_rot2), int(point_up_rot2)],
                                                    [int(x_max_rot3), int(point_down_rot3)],
                                                    [int(x_min_rot4), int(point_down_rot4)]]))

                textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                                [int(x_max), int(point_up)],
                                                [int(x_max), int(point_down)],
                                                [int(x_min), int(point_down)]]))


        return peaks, textline_boxes_rot
    
    def return_rotated_contours(self,slope,img_patch):
            dst = self.rotate_image(img_patch, slope)
            dst = dst.astype(np.uint8)
            dst = dst[:, :, 0]
            dst[dst != 0] = 1
            
            imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
            
    def textline_contours_postprocessing(self, textline_mask, slope, contour_text_interest, box_ind):
        

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255
        textline_mask = textline_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, kernel)
        textline_mask = cv2.erode(textline_mask, kernel, iterations=2)
        
        try:

            dst = self.rotate_image(textline_mask, slope)
            dst = dst[:, :, 0]
            dst[dst != 0] = 1

            contour_text_copy = contour_text_interest.copy()

            contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[
                0]
            contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]

            img_contour = np.zeros((box_ind[3], box_ind[2], 3))
            img_contour = cv2.fillPoly(img_contour, pts=[contour_text_copy], color=(255, 255, 255))


 
            img_contour_rot = self.rotate_image(img_contour, slope)

            img_contour_rot = img_contour_rot.astype(np.uint8)
            imgrayrot = cv2.cvtColor(img_contour_rot, cv2.COLOR_BGR2GRAY)
            _, threshrot = cv2.threshold(imgrayrot, 0, 255, 0)
            contours_text_rot, _ = cv2.findContours(threshrot.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            len_con_text_rot = [len(contours_text_rot[ib]) for ib in range(len(contours_text_rot))]
            ind_big_con = np.argmax(len_con_text_rot)



            _, contours_rotated_clean = self.seperate_lines(dst, contours_text_rot[ind_big_con], slope)


        except:

            contours_rotated_clean = []

        return contours_rotated_clean

    def textline_contours_to_get_slope_correctly(self, textline_mask, img_patch, contour_interest):

        slope_new = 0  # deskew_images(img_patch)

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255

        textline_mask = textline_mask.astype(np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, self.kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, self.kernel)
        textline_mask = cv2.erode(textline_mask, self.kernel, iterations=1)
        imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        main_contours = self.filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.003)

        textline_maskt = textline_mask[:, :, 0]
        textline_maskt[textline_maskt != 0] = 1

        peaks_point, _ = self.seperate_lines(textline_maskt, contour_interest, slope_new)

        mean_dis = np.mean(np.diff(peaks_point))

        len_x = thresh.shape[1]

        slope_lines = []
        contours_slope_new = []
        for kk in range(len(main_contours)):

            xminh = np.min(main_contours[kk][:, 0])
            xmaxh = np.max(main_contours[kk][:, 0])

            yminh = np.min(main_contours[kk][:, 1])
            ymaxh = np.max(main_contours[kk][:, 1])


            if ymaxh - yminh <= mean_dis and (
                    xmaxh - xminh) >= 0.3 * len_x:  # xminh>=0.05*len_x and xminh<=0.4*len_x and xmaxh<=0.95*len_x and xmaxh>=0.6*len_x:
                contours_slope_new.append(main_contours[kk])

                rows, cols = thresh.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(main_contours[kk], cv2.DIST_L2, 0, 0.01, 0.01)

                slope_lines.append((vy / vx) / np.pi * 180)

            if len(slope_lines) >= 2:

                slope = np.mean(slope_lines)  # slope_true/np.pi*180
            else:
                slope = 999

        else:
            slope = 0

        return slope
    def return_contours_of_image(self,image_box_tabels_1):
        
        image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)
        image_box_tabels=image_box_tabels.astype(np.uint8)
        imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours,hierachy
    
    def find_contours_mean_y_diff(self,contours_main):
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        return np.mean( np.diff( np.sort( np.array(cy_main) ) ) )
    
    
    def isNaN(self,num):
        return num != num
    
    def find_num_col(self,regions_without_seperators,sigma_,multiplier=3.8 ):
        regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=1)

        meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

        last_nonzero=len(regions_without_seperators_0)-last_nonzero


        y=regions_without_seperators_0#[first_nonzero:last_nonzero]

        y_help=np.zeros(len(y)+20)

        y_help[10:len(y)+10]=y

        x=np.array( range(len(y)) )




        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+20)

        zneg[10:len(zneg_rev)+10]=zneg_rev

        z=gaussian_filter1d(y, sigma_)
        zneg= gaussian_filter1d(zneg, sigma_)


        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg=peaks_neg-10-10

        

        last_nonzero=last_nonzero-0#100
        first_nonzero=first_nonzero+0#+100

        peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]
        
        peaks=peaks[(peaks>.06*regions_without_seperators.shape[1]) & (peaks<0.94*regions_without_seperators.shape[1])]

        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]

        interest_neg=z[peaks_neg]
        
        
        if interest_neg[0]<0.1:
            interest_neg=interest_neg[1:]
        if interest_neg[len(interest_neg)-1]<0.1:
            interest_neg=interest_neg[:len(interest_neg)-1]
            
        
        
        min_peaks_pos=np.min(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        

        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg#[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg#[(interest_neg<grenze)]
        interest_neg_fin=interest_neg#[(interest_neg<grenze)]

        num_col=(len(interest_neg_fin))+1


        p_l=0
        p_u=len(y)-1
        p_m=int(len(y)/2.)
        p_g_l=int(len(y)/3.)
        p_g_u=len(y)-int(len(y)/3.)
        
        
        diff_peaks=np.abs( np.diff(peaks_neg_fin) )
        diff_peaks_annormal=diff_peaks[diff_peaks<30]
        

        return interest_neg_fin
    def return_deskew_slop(self,img_patch_org,sigma_des):
        img_int=np.zeros((img_patch_org.shape[0],img_patch_org.shape[1]))
        img_int[:,:]=img_patch_org[:,:]#img_patch_org[:,:,0]

        img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(1.2) ) ))
        
        img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(.1)):int( img_int.shape[1]*(.1))+img_int.shape[1] ]=img_int[:,:]
        angels=np.linspace(-4,4,60)

        res=[]
        index_cor=[]
        indexer=0
        for rot in angels:
            img_rot=self.rotate_image(img_resized,rot)
            img_rot[img_rot!=0]=1
            res_me=np.mean(self.find_num_col(img_rot,sigma_des,200.3  ))
            if self.isNaN(res_me):
                pass
            else:
                res.append( res_me )
                index_cor.append(indexer)
            indexer=indexer+1


        res=np.array(res)
        arg_int=np.argmin(res)
        arg_fin=index_cor[arg_int]
        ang_int=angels[arg_fin]
        
        img_rot=self.rotate_image(img_resized,ang_int)
        img_rot[img_rot!=0]=1

        return ang_int

        
    def do_work_of_slopes(self,q,poly,box_sub,boxes_per_process,textline_mask_tot,contours_per_process):
        slope_biggest=0
        slopes_sub = []
        boxes_sub_new=[]
        poly_sub=[]
        for mv in range(len(boxes_per_process)):
            
            
            crop_img, _ = self.crop_image_inside_box(boxes_per_process[mv],
                                                                        np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img=crop_img[:,:,0]
            crop_img=cv2.erode(crop_img,self.kernel,iterations = 2)
            
            try:
                textline_con,hierachy=self.return_contours_of_image(crop_img)
                textline_con_fil=self.filter_contours_area_of_image(crop_img,textline_con,hierachy,max_area=1,min_area=0.0008)
                y_diff_mean=self.find_contours_mean_y_diff(textline_con_fil)

                sigma_des=int(  y_diff_mean * (4./40.0) )

                if sigma_des<1:
                    sigma_des=1

                crop_img[crop_img>0]=1
                slope_corresponding_textregion=self.return_deskew_slop(crop_img,sigma_des)

                
            except:
                slope_corresponding_textregion=999
                
        
            if np.abs(slope_corresponding_textregion)>12.5 and slope_corresponding_textregion!=999:
                slope_corresponding_textregion=slope_biggest
            elif slope_corresponding_textregion==999:
                slope_corresponding_textregion=slope_biggest
            slopes_sub.append(slope_corresponding_textregion)
            
            cnt_clean_rot = self.textline_contours_postprocessing(crop_img
                                                                                        , slope_corresponding_textregion,
                                                                                        contours_per_process[mv], boxes_per_process[mv])
            
            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv] )
            

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new )

    def get_slopes_and_deskew(self, contours,textline_mask_tot):

        slope_biggest=0#self.return_deskew_slop(img_int_p,sigma_des)
        
        num_cores = cpu_count()
        q = Queue()
        poly=Queue()
        box_sub=Queue()
        
        processes = []
        nh=np.linspace(0, len(self.boxes), num_cores+1)
        
        
        for i in range(num_cores):
            boxes_per_process=self.boxes[int(nh[i]):int(nh[i+1])]
            contours_per_process=contours[int(nh[i]):int(nh[i+1])]
            processes.append(Process(target=self.do_work_of_slopes, args=(q,poly,box_sub,  boxes_per_process, textline_mask_tot, contours_per_process)))
        
        for i in range(num_cores):
            processes[i].start()
            
        self.slopes = []
        self.all_found_texline_polygons=[]
        self.boxes=[]
        
        for i in range(num_cores):
            slopes_for_sub_process=q.get(True)
            boxes_for_sub_process=box_sub.get(True)
            polys_for_sub_process=poly.get(True)
            
            for j in range(len(slopes_for_sub_process)):
                self.slopes.append(slopes_for_sub_process[j])
                self.all_found_texline_polygons.append(polys_for_sub_process[j])
                self.boxes.append(boxes_for_sub_process[j])
                
        for i in range(num_cores):
            processes[i].join()
            
        
    def order_of_regions(self, textline_mask,contours_main):
        mada_n=textline_mask.sum(axis=1)
        y=mada_n[:]

        y_help=np.zeros(len(y)+40)
        y_help[20:len(y)+20]=y
        x=np.array( range(len(y)) )


        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        

        sigma_gaus=8

        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)


        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg=peaks_neg-20-20
        peaks=peaks-20
        

        
        if contours_main!=None:
            areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
            x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
            x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

            y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
            y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])


        
        
        if contours_main!=None:
            indexer_main=np.array(range(len(contours_main)))

        
        if contours_main!=None:
            len_main=len(contours_main)
        else:
            len_main=0

        
        matrix_of_orders=np.zeros((len_main,5))
        
        matrix_of_orders[:,0]=np.array( range( len_main ) )
        
        matrix_of_orders[:len_main,1]=1
        matrix_of_orders[len_main:,1]=2
        
        matrix_of_orders[:len_main,2]=cx_main
        matrix_of_orders[:len_main,3]=cy_main

        matrix_of_orders[:len_main,4]=np.array( range( len_main ) )

        peaks_neg_new=[]
        peaks_neg_new.append(0)
        for iii in range(len(peaks_neg)):
            peaks_neg_new.append(peaks_neg[iii])
        peaks_neg_new.append(textline_mask.shape[0])
        
        final_indexers_sorted=[]
        for i in range(len(peaks_neg_new)-1):
            top=peaks_neg_new[i]
            down=peaks_neg_new[i+1]
            
            indexes_in=matrix_of_orders[:,0][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            cxs_in=matrix_of_orders[:,2][(matrix_of_orders[:,3]>=top) & ((matrix_of_orders[:,3]<down))]
            
            sorted_inside=np.argsort(cxs_in)
            
            ind_in_int=indexes_in[sorted_inside]
            
            for j in range(len(ind_in_int)):
                final_indexers_sorted.append(int(ind_in_int[j]) )
        
        return final_indexers_sorted, matrix_of_orders

            

    
    def order_and_id_of_texts(self, found_polygons_text_region ,matrix_of_orders ,indexes_sorted ):
        id_of_texts=[]
        order_of_texts=[]
        index_b=0
        for mm in range(len(found_polygons_text_region)):
            id_of_texts.append('r'+str(index_b) )
            index_matrix=matrix_of_orders[:,0][( matrix_of_orders[:,1]==1 ) & ( matrix_of_orders[:,4]==mm ) ]
            order_of_texts.append(np.where(indexes_sorted == index_matrix)[0][0])

            index_b+=1
            
        order_of_texts
        return order_of_texts, id_of_texts
    
    def write_into_page_xml(self,contours,page_coord,dir_of_image,order_of_texts , id_of_texts):

        found_polygons_text_region=contours


        # create the file structure
        data = ET.Element('PcGts')

        data.set('xmlns',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")
        data.set('xsi:schemaLocation',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")



        metadata=ET.SubElement(data,'Metadata')

        author=ET.SubElement(metadata, 'Creator')
        author.text = 'SBB_QURATOR'


        created=ET.SubElement(metadata, 'Created')
        created.text = datetime.datetime.now().isoformat()
        changetime=ET.SubElement(metadata, 'LastChange')
        changetime.text = datetime.datetime.now().isoformat()


        page=ET.SubElement(data,'Page')

        page.set('imageFilename', self.image_dir)
        page.set('imageHeight',str(self.height_org) ) 
        page.set('imageWidth',str(self.width_org) )
        page.set('type',"content")
        page.set('readingDirection',"left-to-right")
        page.set('textLineOrder',"top-to-bottom" )


        
        page_print_sub=ET.SubElement(page, 'PrintSpace')
        coord_page = ET.SubElement(page_print_sub, 'Coords')
        points_page_print=''

        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm])==2:
                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][0])/self.scale_x ) )
                points_page_print=points_page_print+','
                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][1])/self.scale_y ) )
            else:
                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][0])/self.scale_x) )
                points_page_print=points_page_print+','
                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][1])/self.scale_y) )

            if lmm<(len(self.cont_page[0])-1):
                points_page_print=points_page_print+' '
        coord_page.set('points',points_page_print)
        

        if len(contours)>0:
            region_order=ET.SubElement(page, 'ReadingOrder')
            region_order_sub = ET.SubElement(region_order, 'OrderedGroup')
            
            region_order_sub.set('id',"ro357564684568544579089")
    
            args_sort=np.argsort(order_of_texts)
            for vj in args_sort:
                name="coord_text_"+str(vj)
                name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
                name.set('index',str(order_of_texts[vj]) )
                name.set('regionRef',id_of_texts[vj])
    
    
            id_indexer=0
            id_indexer_l=0
    
            for mm in range(len(found_polygons_text_region)):
                textregion=ET.SubElement(page, 'TextRegion')
    
                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1
                
                textregion.set('type','paragraph')
                #if mm==0:
                #    textregion.set('type','heading')
                #else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                
                points_co=''
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_text_region[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_text_region[mm][lmm][0][1] +page_coord[0])/self.scale_y) )
    
                    if lmm<(len(found_polygons_text_region[mm])-1):
                        points_co=points_co+' '
                #print(points_co)
                coord_text.set('points',points_co)
                
                
                
                for j in range(len(self.all_found_texline_polygons[mm])):
    
                    textline=ET.SubElement(textregion, 'TextLine')
                    
                    textline.set('id','l'+str(id_indexer_l))
                    
                    id_indexer_l+=1
                    
    
                    coord = ET.SubElement(textline, 'Coords')
                    #points = ET.SubElement(coord, 'Points') 
    
                    points_co=''
                    for l in range(len(self.all_found_texline_polygons[mm][j])):
                        #point = ET.SubElement(coord, 'Point') 
    
    
    
                        #point.set('x',str(found_polygons[j][l][0]))  
                        #point.set('y',str(found_polygons[j][l][1]))
                        if len(self.all_found_texline_polygons[mm][j][l])==2:
                            points_co=points_co+str( int( (self.all_found_texline_polygons[mm][j][l][0] +page_coord[2]
                                                    +self.all_box_coord[mm][2])/self.scale_x) )
                            points_co=points_co+','
                            points_co=points_co+str( int( (self.all_found_texline_polygons[mm][j][l][1] +page_coord[0]
                                                    +self.all_box_coord[mm][0])/self.scale_y) )
                        else:
                            points_co=points_co+str( int( ( self.all_found_texline_polygons[mm][j][l][0][0] +page_coord[2]
                                                    +self.all_box_coord[mm][2])/self.scale_x ) )
                            points_co=points_co+','
                            points_co=points_co+str( int( ( self.all_found_texline_polygons[mm][j][l][0][1] +page_coord[0]
                                                    +self.all_box_coord[mm][0])/self.scale_y) ) 
    
                        if l<(len(self.all_found_texline_polygons[mm][j])-1):
                            points_co=points_co+' '
                    #print(points_co)
                    coord.set('points',points_co)



        tree = ET.ElementTree(data)
        tree.write(os.path.join(self.dir_out, self.f_name) + ".xml")
 
    
    def run(self):
        
        #get image and sclaes, then extract the page of scanned image
        t1=time.time()
        self.get_image_and_scales()
        image_page,page_coord=self.extract_page()

        
        ##########  
        K.clear_session()
        gc.collect()
        t2=time.time()
        
        
        # extract text regions and corresponding contours and surrounding box
        text_regions=self.extract_text_regions(image_page)

        contours=self.get_text_region_contours_and_boxes(text_regions)
        

        
        ##########  
        K.clear_session()
        gc.collect()
        
        t3=time.time()

        
        if len(contours)>0:
            

            
            # extracting textlines using segmentation
            textline_mask_tot=self.textline_contours(image_page)
            #print(textline_mask_tot)
            #plt.imshow(textline_mask_tot)
            #plt.show()
            ##########  
            K.clear_session()
            gc.collect()
            
            t4=time.time()
            
            # get orders of each textregion. This method by now only works for one column documents. 
            indexes_sorted, matrix_of_orders=self.order_of_regions(textline_mask_tot,contours)
            order_of_texts, id_of_texts=self.order_and_id_of_texts(contours ,matrix_of_orders ,indexes_sorted )
            
            ##########  
            gc.collect()
            
            t5=time.time()
 
            # just get the textline result for each box of text regions
            #self.get_textlines_for_each_textregions(textline_mask_tot)
            
            ##########  

            
            # calculate the slope for deskewing for each box of text region.
            self.get_slopes_and_deskew(contours,textline_mask_tot)
            
            
            ##########  
            gc.collect()
            
    
            t6=time.time()
            
            # do deskewing for each box of text region.
            ###self.deskew_textline_patches(contours,textline_mask_tot)
            
            self.get_all_image_patches_coordination(image_page)
            
            ########## 
            ##########  
            gc.collect()
            
            t7=time.time()

        else:
            contours=[]
            order_of_texts=None
            id_of_texts=None
        self.write_into_page_xml(contours,page_coord,self.dir_out , order_of_texts , id_of_texts)

        # Destroy the current Keras session/graph to free memory
        K.clear_session()
        
        print( "time total = "+"{0:.2f}".format(time.time()-t1) )
        print( "time needed for page extraction = "+"{0:.2f}".format(t2-t1) )
        print( "time needed for text region extraction and get contours = "+"{0:.2f}".format(t3-t2) )
        print( "time needed for textlines = "+"{0:.2f}".format(t4-t3) )
        print( "time needed to get order of regions = "+"{0:.2f}".format(t5-t4) )
        print( "time needed to get slopes of regions (deskewing) = "+"{0:.2f}".format(t6-t5) )
        print( "time needed to implement deskewing = "+"{0:.2f}".format(t7-t6) )


        

@click.command()
@click.option('--image', '-i', help='image filename', type=click.Path(exists=True, dir_okay=False))
@click.option('--out', '-o', help='directory to write output xml data', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', help='directory of models', type=click.Path(exists=True, file_okay=False))
def main(image, out, model):
    possibles = globals()  # XXX unused?
    possibles.update(locals())
    x = textlineerkenner(image, out, None, model)
    x.run()


if __name__ == "__main__":
    main()
