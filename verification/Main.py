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
        th1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        img2 = cv2.imread("verification/imgs/"+row[1],0) # trainImage
        th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        blur1 = cv2.blur(th1,(5,5))
        blur2 = cv2.blur(th2,(5,5))
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()


        # find the keypoints and descriptors with SIFT

        kp1, des1 = sift.detectAndCompute(blur1,None)
        kp2, des2 = sift.detectAndCompute(blur2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        try:



            matches = bf.knnMatch(des1,des2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.90*n.distance:
                    good.append([m])

            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
            cv2.imwrite(row[0]+'-'+row[1]+'.png',img3)

            file1 = open("Match.txt","a")
            n1=row[0].split(".jpg")
            n2=row[1].split(".jpg")
            if (len(good)>1):
                file1.write(n1[0]+"-"+n2[0]+"Result:1\n")
                file1.close()
            else:
                file1.write(n1[0]+"-"+n2[0]+" Result:0\n")
                file1.close()
        except:
            print("Descrittore Non trovato")
