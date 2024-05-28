"""
Created on Tue Dec 7 09:10:43 2021

silency map created

@author: Andy
"""

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# Preprocessing: Filter the image with Mean Shift for an initial soft segmentation of the image.
# =============================================================================
def backproject(source, target, levels = 2, scale = 1):
       hsv = cv.cvtColor(source,  cv.COLOR_BGR2HSV)
       hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
       # calculating object histogram
       roihist = cv.calcHist([hsv],[0, 2], None, \
           [levels, levels], [0, 170, 0, 256] )

       # normalize histogram and apply backprojection
       cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
       dst = cv.calcBackProject([hsvt],[0,2],roihist,[0,170,0,256], scale)
       return dst

# =============================================================================
# Find the bounding box of the connected component with the largest area.
# =============================================================================
def largest_contour_rect(saliency):
        contours, hierarchy = cv.findContours(saliency,  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea)
        # print(contours[-1])
        print(cv.boundingRect(contours[-1]))
        return cv.boundingRect(contours[-1])


# =============================================================================
# enhance contrast or brightness
# =============================================================================
def modify_contrast(img, alpha=1.15):
    array_alpha = np.array([alpha])  # contrast
    array_beta = np.array([0.0])  # brightness
    img = cv.add(img, array_beta)
    img = cv.multiply(img, array_alpha)
    img = np.clip(img, 0, 255)
    return img

# =============================================================================
# Refine the salient region with Grabcut.
# =============================================================================
def refine_saliency_with_grabcut(img, saliency):
        rect = largest_contour_rect(saliency)
        print(rect)
        bgdmodel = np.zeros((1, 65),np.float64)
        fgdmodel = np.zeros((1, 65),np.float64)
        saliency[np.where(saliency > 0)] = cv.GC_FGD
        mask = saliency
        if rect == (0, 0, 750, 500):
            return mask
        else:
            cv.grabCut(img, mask, rect, bgdmodel, fgdmodel, 5, cv.GC_INIT_WITH_RECT)
            mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        return mask


def reverse(grayimg):
    shape = grayimg.shape
    for y in range(shape[1]):
        for x in range(shape[0]):
              grayimg[x][y] = 255 - grayimg[x][y]
    return grayimg


def mask1(img, mask):
    s = img.shape
    w, h = s[1], s[0]

    for y in range(w):
        for x in range(h):
            if mask[x][y] == 255:
                img[x][y][0] = 0
                img[x][y][1] = 0
                img[x][y][2] = 0


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    # cv.imshow('labeled.png', labeled_img)
    return labeled_img

def dilation(img, long=3):
    kernal = np.ones((long, long), np.uint8)
    dila = cv.dilate(img, kernal, iterations=1)
    return dila


def eroding(img, long=3):
    kernal = np.ones((long, long), np.uint8)
    erod = cv.erode(img, kernal, iterations=1)
    return erod


def teethcounter(img):

    reimg = cv.resize(img, (750,500), interpolation=cv.INTER_AREA)
    reimg3 = reimg.copy()

    reimg2 = reimg.copy()

    cv.pyrMeanShiftFiltering(reimg, 2, 10, reimg, 2)  # meanshift smoothing
    reimg = modify_contrast(reimg)  # 加強對比
    reimg = eroding(reimg)
    reimg = dilation(reimg)

    backproj = np.uint8(backproject(reimg, reimg, levels=2))

    cv.normalize(backproj,backproj,0,255,cv.NORM_MINMAX)
    saliencies = [backproj, backproj, backproj]
    saliency = cv.merge(saliencies)
    cv.pyrMeanShiftFiltering(saliency, 20, 100, saliency, 2)
    saliency = cv.cvtColor(saliency, cv.COLOR_BGR2GRAY)
    cv.equalizeHist(saliency, saliency)
    T, saliency = cv.threshold(saliency, 100, 255, cv.THRESH_BINARY)

    saliency = refine_saliency_with_grabcut(reimg2, saliency)

    saliency = reverse(saliency)  # 得到反轉的saliency map

    mask1(reimg2, saliency)  # 將mask是黑色的地方圖黑到img

    gray = cv.cvtColor(reimg2, cv.COLOR_RGB2GRAY)
    kernel_size = 3
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    low_threshold = 5
    high_threshold = 50
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
    result = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel=(3, 3), iterations=4)

    num_labels, labels_img = cv.connectedComponents(result, connectivity=8, ltype=None)

    outimg = imshow_components(labels_img)
    return outimg

# =======================================================================
# main project
# ========================================================================
def main():
    # img = cv.imread('./NTU_Dentist/IMG_7630.JPG')
    # size = img.shape
    # height, width = size[0], size[1]
    # teethcounter(img)
    path = './NTU_Dentist/original'
    filelist = os.listdir(path)
    for filename in filelist:
        print(filename)
        img = cv.imread(path + '/' + filename)
        output_img = teethcounter(img)
        cv.imwrite('./output/original/' + filename, output_img)

main()

# ===============================
# # sobel
# x = cv.Sobel(reimg2, cv.CV_16S, 1, 0)
# y = cv.Sobel(reimg2, cv.CV_16S, 0, 1)
# absX = cv.convertScaleAbs(x)
# absY = cv.convertScaleAbs(y)
# edges = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
# ===============================

# showimg
# cv.imshow('img', saliency)
# cv.imshow('img2', edges)
# cv.imshow('img4', labels_img)
# cv.imshow('img3', reimg2)
# cv.imshow('img5', reimg3)
# cv.waitKey(0)