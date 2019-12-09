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
    tool to extract text lines from document images
    """


class textline_detector:
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
        self.model_page_dir = dir_models + '/model_page_new.h5'
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

    def filter_contours_area_of_image(self, image, contours, hierarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(
                    image.shape[:2]) and hierarchy[0][jv][3] == -1 :  # and hierarchy[0][jv][3]==-1 :
                found_polygons_early.append(
                    np.array([ [point] for point in polygon.exterior.coords], dtype=np.uint))
            jv += 1
        return found_polygons_early

    def filter_contours_area_of_image_interiors(self, image, contours, hierarchy, max_area, min_area):
        found_polygons_early = list()

        jv = 0
        for c in contours:
            if len(c) < 3:  # A polygon cannot have less than 3 points
                continue

            polygon = geometry.Polygon([point[0] for point in c])
            area = polygon.area
            if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and \
                    hierarchy[0][jv][3] != -1:
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
            self.img_hight_int = int(self.image.shape[0]*1.1)
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 3300 and self.image.shape[0] >= 2000:
            self.img_hight_int = int(self.image.shape[0]*1.1)
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))

        elif self.image.shape[0] < 4000 and self.image.shape[0] >= 3300 and self.image.shape[1]<2400 :
            self.img_hight_int = int(self.image.shape[0]*1.1)# 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
            
        elif self.image.shape[0] < 4000 and self.image.shape[0] >= 3300 and self.image.shape[1]>=2400 :
            self.img_hight_int = 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
            
        elif self.image.shape[0] < 5400 and self.image.shape[0] > 4000 and self.image.shape[1]>3300 :
            self.img_hight_int = int(self.image.shape[0]*1.6)# 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        elif self.image.shape[0] < 11000 and self.image.shape[0] >= 7000 :
            self.img_hight_int = int(self.image.shape[0]*1.6)# 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
        else:
            self.img_hight_int = int(self.image.shape[0]*1.1)# 6500
            self.img_width_int = int(self.img_hight_int * self.image.shape[1] / float(self.image.shape[0]))
            #self.img_hight_int = self.image.shape[0]
            #self.img_width_int = self.image.shape[1]

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
        #for ii in range(1):
        #    img = cv2.GaussianBlur(img, (15, 15), 0)

        
        img_page_prediction=self.do_prediction(patches,img,model_page)
        
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=6)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)
        
        try:
            box = [x, y, w, h]
            
            croped_page, page_coord = self.crop_image_inside_box(box, self.image)
            

            self.cont_page=[]
            self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] , 
                                                        [ page_coord[3] , page_coord[0] ] ,
                                                        [ page_coord[3] , page_coord[1] ] ,
                                                    [ page_coord[2] , page_coord[1] ]] ) )
        except:
            box = [0, 0, self.image.shape[1]-1, self.image.shape[0]-1]
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

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contours = self.filter_contours_area_of_image(thresh, contours, hierarchy, max_area=1, min_area=0.00001)
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

    def seperate_lines(self, img_patch, contour_text_interest, thetha):
        (h, w) = img_patch.shape[:2]
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
        x_max_cont = img_patch.shape[1]
        y_min_cont = 0
        y_max_cont = img_patch.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        textline_patch_sum_along_width = img_patch.sum(axis=1)

        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
        y_padded = np.zeros(len(y) + 40)
        y_padded[20:len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real)<=2 and len(peaks_real)>1:
            sigma_gaus=10
        else:
            sigma_gaus=8
    
    
        y_padded_smoothed= gaussian_filter1d(y_padded, sigma_gaus)
        y_padded_up_to_down=-y_padded+np.max(y_padded)
        y_padded_up_to_down_padded=np.zeros(len(y_padded_up_to_down)+40)
        y_padded_up_to_down_padded[20:len(y_padded_up_to_down)+20]=y_padded_up_to_down
        y_padded_up_to_down_padded= gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)
        

        peaks, _ = find_peaks(y_padded_smoothed, height=0)
        peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)
        
        mean_value_of_peaks=np.mean(y_padded_smoothed[peaks])
        std_value_of_peaks=np.std(y_padded_smoothed[peaks])
        peaks_values=y_padded_smoothed[peaks]
        

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
            #print('11')
            for jj in range(len(peaks)):
                
                if jj==(len(peaks)-1):
                    dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                    dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                    
                    if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                        point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down =y_max_cont-1##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                    else:
                        point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down =y_max_cont-1##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                    point_down_narrow = peaks[jj] + first_nonzero + int(
                        1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)
                else:
                    dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                    dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                    
                    if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                        point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                    else:
                        point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                        point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                    point_down_narrow = peaks[jj] + first_nonzero + int(
                        1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)



                if point_down_narrow >= img_patch.shape[0]:
                    point_down_narrow = img_patch.shape[0] - 2

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
                    point_up = 0#peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                    if point_up < 0:
                        point_up = 1
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
                elif jj == 1:
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
                    if point_down >= img_patch.shape[0]:
                        point_down = img_patch.shape[0] - 2
                    point_up = peaks[jj] + first_nonzero - int(1. / 1.8 * dis_to_next)

                distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True)
                             for mj in range(len(xv))]
                distances = np.array(distances)

                xvinside = xv[distances >= 0]

                if len(xvinside) == 0:
                    x_min = x_min_cont
                    x_max = x_max_cont
                else:
                    x_min = np.min(xvinside)
                    x_max = np.max(xvinside)

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
                    point_down = peaks[jj] + first_nonzero + int(1. / 1.7 * dis_to_next)
                    if point_down >= img_patch.shape[0]:
                        point_down = img_patch.shape[0] - 2
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


    def return_contours_of_image(self,image_box_tabels_1):
        
        image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)
        image_box_tabels=image_box_tabels.astype(np.uint8)
        imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours,hierarchy
    
    def find_contours_mean_y_diff(self,contours_main):
        M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
        return np.mean( np.diff( np.sort( np.array(cy_main) ) ) )
    
    
    def isNaN(self,num):
        return num != num
    
    def get_standard_deviation_of_summed_textline_patch_along_width(self,img_patch,sigma_,multiplier=3.8 ):
        img_patch_sum_along_width=img_patch[:,:].sum(axis=1)

        img_patch_sum_along_width_updown=img_patch_sum_along_width[len(img_patch_sum_along_width)::-1]

        first_nonzero=(next((i for i, x in enumerate(img_patch_sum_along_width) if x), 0))
        last_nonzero=(next((i for i, x in enumerate(img_patch_sum_along_width_updown) if x), 0))

        last_nonzero=len(img_patch_sum_along_width)-last_nonzero


        y=img_patch_sum_along_width#[first_nonzero:last_nonzero]

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
        
        interest_pos=z[peaks]
        
        interest_pos=interest_pos[interest_pos>10]
        
        interest_neg=z[peaks_neg]
        
        min_peaks_pos=np.mean(interest_pos)
        min_peaks_neg=0#np.min(interest_neg)
        
        dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
        #print(interest_pos)
        grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        peaks_neg_fin=peaks_neg[(interest_neg<grenze)]
        interest_neg_fin=interest_neg[(interest_neg<grenze)]
        
        return interest_neg_fin,np.std(z)
    
    def return_deskew_slope(self,img_patch,sigma_des):
        img_patch_copy=np.zeros((img_patch.shape[0],img_patch.shape[1]))
        img_patch_copy[:,:]=img_patch[:,:]#img_patch_org[:,:,0]

        img_patch_padded=np.zeros((int( img_patch_copy.shape[0]*(1.2) ) , int( img_patch_copy.shape[1]*(2.6) ) ))
        
        img_patch_padded[ int( img_patch_copy.shape[0]*(.1)):int( img_patch_copy.shape[0]*(.1))+img_patch_copy.shape[0] , int( img_patch_copy.shape[1]*(.8)):int( img_patch_copy.shape[1]*(.8))+img_patch_copy.shape[1] ]=img_patch_copy[:,:]
        angles=np.linspace(-12,12,40)

        res=[]
        num_of_peaks=[]
        index_cor=[]
        var_res=[]
        
        indexer=0
        for rot in angles:
            img_rotated=self.rotate_image(img_patch_padded,rot)
            img_rotated[img_rotated!=0]=1
            try:
                neg_peaks,var_spectrum=self.get_standard_deviation_of_summed_textline_patch_along_width(img_rotated,sigma_des,20.3  )
                res_me=np.mean(neg_peaks)
                if res_me==0:
                    res_me=1000000000000000000000
                else:
                    pass
                    
                res_num=len(neg_peaks)
            except:
                res_me=1000000000000000000000
                res_num=0
                var_spectrum=0
            if self.isNaN(res_me):
                pass
            else:
                res.append( res_me )
                var_res.append(var_spectrum)
                num_of_peaks.append( res_num )
                index_cor.append(indexer)
            indexer=indexer+1


        try:
            var_res=np.array(var_res)
            
            ang_int=angles[np.argmax(var_res)]#angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
        except:
            ang_int=0

        return ang_int


    def do_work_of_slopes(self,queue_of_slopes_per_textregion,queue_of_textlines_rectangle_per_textregion
                          ,queue_of_textregion_box,boxes_per_process,queue_of_quntours_of_textregion,textline_mask_tot,contours_per_process):
        
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess=[]
        textlines_rectangles_per_each_subprocess=[]
        contours_textregion_per_each_subprocess=[]

        for mv in range(len(boxes_per_process)):
            
            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            crop_img, _ = self.crop_image_inside_box(boxes_per_process[mv],
                                                                        np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img=crop_img[:,:,0]
            crop_img=cv2.erode(crop_img,self.kernel,iterations = 2)
            
            try:
                sigma_des=2
                slope_corresponding_textregion=self.return_deskew_slope(crop_img,sigma_des)
            except:
                slope_corresponding_textregion=999
                
        
            if np.abs(slope_corresponding_textregion)>12.5 and slope_corresponding_textregion!=999:
                slope_corresponding_textregion=0
            elif slope_corresponding_textregion==999:
                slope_corresponding_textregion=0
            slopes_per_each_subprocess.append(slope_corresponding_textregion)
            
            bounding_rectangle_of_textlines = self.textline_contours_postprocessing(crop_img
                                                                                        , slope_corresponding_textregion,
                                                                                        contours_per_process[mv], boxes_per_process[mv])
            
            textlines_rectangles_per_each_subprocess.append(bounding_rectangle_of_textlines)
            bounding_box_of_textregion_per_each_subprocess.append(boxes_per_process[mv] )

            

        queue_of_slopes_per_textregion.put(slopes_per_each_subprocess)
        queue_of_textlines_rectangle_per_textregion.put(textlines_rectangles_per_each_subprocess)
        queue_of_textregion_box.put(bounding_box_of_textregion_per_each_subprocess )
        queue_of_quntours_of_textregion.put(contours_textregion_per_each_subprocess)

    def get_slopes_and_deskew(self, contours,textline_mask_tot):
        num_cores = cpu_count()
        
        queue_of_slopes_per_textregion = Queue()
        queue_of_textlines_rectangle_per_textregion=Queue()
        queue_of_textregion_box=Queue()
        queue_of_quntours_of_textregion=Queue()
        
        processes = []
        nh=np.linspace(0, len(self.boxes), num_cores+1)
        
        
        for i in range(num_cores):
            boxes_per_process=self.boxes[int(nh[i]):int(nh[i+1])]
            contours_per_process=contours[int(nh[i]):int(nh[i+1])]
            processes.append(Process(target=self.do_work_of_slopes, args=(queue_of_slopes_per_textregion,queue_of_textlines_rectangle_per_textregion,
                                                                          queue_of_textregion_box,  boxes_per_process, queue_of_quntours_of_textregion, textline_mask_tot, contours_per_process)))
        
        for i in range(num_cores):
            processes[i].start()
            
        self.slopes = []
        self.all_found_texline_polygons=[]
        all_found_text_regions=[]
        self.boxes=[]
        
        for i in range(num_cores):
            slopes_for_sub_process=queue_of_slopes_per_textregion.get(True)
            boxes_for_sub_process=queue_of_textregion_box.get(True)
            polys_for_sub_process=queue_of_textlines_rectangle_per_textregion.get(True)
            contours_for_subprocess=queue_of_quntours_of_textregion.get(True)
            
            for j in range(len(slopes_for_sub_process)):
                self.slopes.append(slopes_for_sub_process[j])
                self.all_found_texline_polygons.append(polys_for_sub_process[j])
                self.boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                
        for i in range(num_cores):
            processes[i].join()
            
        return all_found_text_regions
            
        
    def order_of_regions(self, textline_mask,contours_main):
        textline_sum_along_width=textline_mask.sum(axis=1)
        
        y=textline_sum_along_width[:]
        y_padded=np.zeros(len(y)+40)
        y_padded[20:len(y)+20]=y
        x=np.array( range(len(y)) )


        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        

        sigma_gaus=8

        z= gaussian_filter1d(y_padded, sigma_gaus)
        zneg_rev=-y_padded+np.max(y_padded)

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

        data.set('xmlns',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")
        data.set('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")
        data.set('xsi:schemaLocation',"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")



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
        
        #get image and scales, then extract the page of scanned image
        t1=time.time()
        self.get_image_and_scales()
        image_page,page_coord=self.extract_page()

        
        ##########  
        K.clear_session()
        gc.collect()
        t2=time.time()
        
        
        # extract text regions and corresponding contours and surrounding box
        text_regions=self.extract_text_regions(image_page)
        
        text_regions = cv2.erode(text_regions, self.kernel, iterations=3)
        text_regions = cv2.dilate(text_regions, self.kernel, iterations=4)
        
        #plt.imshow(text_regions[:,:,0])
        #plt.show()

        contours=self.get_text_region_contours_and_boxes(text_regions)
        

        
        ##########  
        K.clear_session()
        gc.collect()
        
        t3=time.time()

        
        if len(contours)>0:
            

            
            # extracting textlines using segmentation
            textline_mask_tot=self.textline_contours(image_page)
            ##########  
            K.clear_session()
            gc.collect()
            
            t4=time.time()
            
            
            # calculate the slope for deskewing for each box of text region.
            contours=self.get_slopes_and_deskew(contours,textline_mask_tot)
            
            gc.collect()
            t5=time.time()
            
            
            # get orders of each textregion. This method by now only works for one column documents. 
            indexes_sorted, matrix_of_orders=self.order_of_regions(textline_mask_tot,contours)
            order_of_texts, id_of_texts=self.order_and_id_of_texts(contours ,matrix_of_orders ,indexes_sorted )
            
            
            ##########  
            gc.collect()
            t6=time.time()
            
            
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
        if len(contours)>0:
            print( "time needed for textlines = "+"{0:.2f}".format(t4-t3) )
            print( "time needed to get slopes of regions (deskewing) = "+"{0:.2f}".format(t5-t4) )
            print( "time needed to get order of regions = "+"{0:.2f}".format(t6-t5) )
            print( "time needed to implement deskewing = "+"{0:.2f}".format(t7-t6) )

        

@click.command()
@click.option('--image', '-i', help='image filename', type=click.Path(exists=True, dir_okay=False))
@click.option('--out', '-o', help='directory to write output xml data', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', help='directory of models', type=click.Path(exists=True, file_okay=False))
def main(image, out, model):
    possibles = globals()  # XXX unused?
    possibles.update(locals())
    x = textline_detector(image, out, None, model)
    x.run()


if __name__ == "__main__":
    main()
 
