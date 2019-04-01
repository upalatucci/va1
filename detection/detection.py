import cv2
import csv
import sys

def read_prot(file_name):
    res = list()
    names = list()
    classes = dict()
    j = 0
    sift = cv2.xfeatures2d.SIFT_create()
    with open(file_name) as csvfile:
        spam_reader = csv.reader(csvfile, delimiter=',')
        for row in spam_reader:
            names.append(row[0])
            index = int(float(row[1]))
            classes[j]=index
            j += 1
            img = cv2.imread('./prot/' + row[0],0)
            kp1, des1 = sift.detectAndCompute(img, None)
            res.append(des1)
    return res, names, classes

def read_test(file_name):
    res = list()
    names = list()
    sift = cv2.xfeatures2d.SIFT_create()
    with open(file_name) as csvfile:
        spam_reader = csv.reader(csvfile, delimiter=',')
        for row in spam_reader:
            names.append(row[0])
            img = cv2.imread('./test/' + row[0],0)
            kp1, des1 = sift.detectAndCompute(img, None)
            res.append(des1)
    return res, names

def classification(img1,img2,bf):
    try:
        points=0
        if img1 is not None and img2 is not None:
            matches = bf.match(img1, img2)
            matches = sorted(matches, key=lambda x: x.distance)
            for m in matches:
                if m.distance < 135:
                    points += (135-m.distance)
        return points
    except Exception as ex:
        print(ex)

def write(result, names_test,name_file):
    with open(name_file, mode='w') as csvfile:
        spam_writer=csv.writer(csvfile,delimiter=',')
        for i in names_test:
            spam_writer.writerow([result[i]])


if __name__=='__main__':
    list_prot,names_prot,classes_prot=read_prot('prot.csv')
    list_test,names_test=read_test(sys.argv[1])
    result={i:0 for i in names_test}
    bf = cv2.BFMatcher()
    for i in range(len(list_test)):
        res = list()
        val_max=0
        val_index=0
        for j in range(len(names_prot)):
            val = classification(list_test[i], list_prot[j], bf)
            if val>val_max:
                val=val_max
                val_index=j
        index = names_test[i]
        result[index] = classes_prot[val_index]
    write(result,names_test,sys.argv[2])

