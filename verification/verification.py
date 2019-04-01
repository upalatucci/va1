import cv2
import sys
import csv


def verification(nome_file_test,nome_file_risultati):

    tot_match = 0
    correct_match=0
    with open(nome_file_test, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in spamreader:
            i=i+1
            #binary_correct_value = row[2]

            img1 = cv2.imread("imgs/"+row[0],0)          # queryImage

            th1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

            img2 = cv2.imread("imgs/"+row[1],0) # trainImage
            th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
            #blur1 = cv2.blur(th1,(5,5))
            #blur2 = cv2.blur(th2,(5,5))
            # Initiate SIFT detector


            sift = cv2.xfeatures2d.SIFT_create()


            # find the keypoints and descriptors with SIFT

            kp1, des1 = sift.detectAndCompute(th1,None)
            kp2, des2 = sift.detectAndCompute(th2,None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()

            try:
                matches = bf.knnMatch(des1,des2, k=2)

                # Apply ratio test
                good = []
                for m,n in matches:
                    if m.distance < 0.8499*n.distance:
                        good.append([m])

                # cv2.drawMatchesKnn expects list of lists as matches.
                #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
                #cv2.imwrite(row[0]+'-'+row[1]+'.png',img3)

                file1 = open(nome_file_risultati,"a")
                n1=row[0].split(".jpg")
                n2=row[1].split(".jpg")
                if (len(good)>1):
                    file1.write(n1[0]+"-"+n2[0]+"Result:1\n")
                    file1.close()
                    #if(float(binary_correct_value)==1.0):
                    #    correct_match=correct_match+1

                else:
                    file1.write(n1[0]+"-"+n2[0]+" Result:0\n")
                    file1.close()
                    #if(float(binary_correct_value)==0.0):
                    #    correct_match=correct_match+1

                #tot_match =tot_match+1
            except Exception as e :
                print("Descrittore Non trovato")

    #print("Accuracy= "+str(correct_match/tot_match*100))


if __name__ == "__main__":
    nome_file_test = sys.argv[1]
    nome_file_risultati = sys.argv[2]

    verification(nome_file_test,nome_file_risultati)
