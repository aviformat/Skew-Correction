""" From your python2.7 console type

>>>python findAngle.py 'image name'

for eg. I want to predict angle of skew correction of image 'xyz.jpg', type

>>>python findAngle.py xyz.jpg

if you also want to analyse angle of your image, just un-comment
line number- 88,89,133,134
"""




import cv2
import numpy as np
import operator
import collections
import math
import sys
import os

size=1024       #converts input image into dimension size*size

def draw_plane(min,max,stride,dim):         #draws protector from angle "min" to "max" with difference of stride and on size (dim*dim)
    img = np.zeros((dim, dim, 3), np.uint8)
    img[:, :, :] = (255, 255, 255)
    j=0
    i=min
    while(i<=max):
        x = float(size/2) / np.tan(np.deg2rad(i))
        cv2.line(img, (int(size/2.0 + x), 0), (size/2, size/2), (0, 0, 0), 1)

        if j%3==0 and i<90:
            cv2.putText(img, "+"+str(90-i), (int(size/2.0+x-40), 0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        if j%3==0 and i>90:
            cv2.putText(img, "-"+str(i-90), (int(size/2.0+x), 0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        j+=1
        i = i + stride
    return img

def rotateImage(image, angle):              #rotate image with "angle"
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate(image,angle):                    #merge rotated image with a white background to avoid cropping
    s_image=rotateImage(image,angle)

    l_image = np.zeros((size, size, 3), np.uint8)

    l_image[:,:,:]=(255,255,255)

    finalimg=l_image+s_image
    return finalimg

def find_Angle(img_str):                       #correct skew from -45 to 45 degree
    img1 = cv2.imread(str(img_str))

    img1 = cv2.resize(img1, (size, size))
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.Canny(img, 100,200, apertureSize=5, L2gradient=True)

    img_b = draw_plane(45, 135, 5, size)

    # cv2.imshow('szxv', img)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, 1)



    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 30, minLineLength=20)


    allAngle = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            angle = (y1 - y2) / float(x2 - x1)
            angle = np.rad2deg(np.arctan(angle))
            if angle >= -45 and angle <= 45:
                allAngle.append(angle)


    img2 = img1
    cv2.line(img2, (size / 2, 0), (size / 2, size), (255, 0, 127), 4)
    cv2.line(img2, (0, size/2), (size, size/2), (0, 0, 127), 4)
    comb_img = cv2.bitwise_and(img2, img_b)

    # cv2.imshow('Original Image', comb_img)            #shows the original image
    # cv2.waitKey(0)

    degree_bin = 3
    bin = []
    for i in range(int(90 / degree_bin) + 1):
        bin.append(-45 + i * degree_bin)


    hist, bins = np.histogram(allAngle, bin)

    maxi = [i for i, j in enumerate(hist) if j == max(hist)]

    if bins[maxi][0] >= 0:
        blah = 1
    else:
        blah = 2

    finalAngle = []
    for ang in allAngle:
        if ang >= bins[maxi[0] - 1] and ang < bins[maxi[0]] and blah == 2:
            finalAngle.append(ang)
        if ang >= bins[maxi[0]] and ang < bins[maxi[0] + 1]:
            finalAngle.append(ang)
        if ang > bins[maxi[0] + 1] and ang <= bins[maxi[0] + 2] and blah == 1:
            finalAngle.append(ang)


    ans = np.mean(finalAngle)


    img_final = rotate(img2, -1 * ans)
    img_final = cv2.resize(img_final, (size, size))

    img_final = cv2.bitwise_and(img_final, img_b)

    cv2.line(img_final, (int(size / 2.0), 0), (size / 2, size), (0, 0, 255), 3)
    cv2.line(img_final, (0, size / 2), (size, size / 2), (0, 255, 0), 3)
    cv2.ellipse(img_final, (size / 2, size / 2), (200, 200), 0, 270, 270 + ans, 255, 3)
    cv2.putText(img_final, str(ans), (int(size * 2 / 3.0), int(size / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 3)

    i = 0
    while (i < size):
        cv2.line(img_final, (0, i), (size, i), (0, 255, 0), 1)
        i += 32
    # cv2.imshow('sdvzv', img_final)            #show skew corrected image with angle
    # cv2.waitKey(0)

    vis = np.concatenate((comb_img, img_final), axis=1)
    nam=os.path.join('results/',str(img_str))
    # print nam
    # cv2.imwrite(nam,vis)
    # cv2.waitKey(0)
    # vis=cv2.resize(vis,(1300,600))
    # cv2.imshow('sdvnjvn',vis)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return -1*ans



# im=sys.argv[1]
# angle=find_Angle(im)
# print angle
