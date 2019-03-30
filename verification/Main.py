import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt

import csv
with open('verification/gt.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in spamreader:
        i=i+1

        img1 = cv2.imread("verification/imgs/"+row[0],0)          # queryImage
        img2 = cv2.imread("verification/imgs/"+row[1],0) # trainImage
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()


        # find the keypoints and descriptors with SIFT

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(des1,des2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.95*n.distance:
                    good.append([m])

            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
            cv2.imwrite('img'+str(i)+'.png',img3)
        except:
            print("Descrittore Non trovato")
