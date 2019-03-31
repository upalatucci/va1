import cv2
import csv
import sys

def read_prot(file_name):
    res = list()
    names = list()
    classes = {i: list() for i in range(19)}
    j = 0
    with open(file_name) as csvfile:
        spam_reader = csv.reader(csvfile, delimiter=',')
        for row in spam_reader:
            names.append(row[0])
            index = int(float(row[1]))
            classes[index].append(j)
            j += 1
            img = cv2.imread('./prot/' + row[0], 0)
            th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)
            res.append(th1)
    return res, names, classes

def read_test(file_name):
    res = list()
    names = list()
    classes = dict()
    with open(file_name) as csvfile:
        spam_reader = csv.reader(csvfile, delimiter=',')
        for row in spam_reader:
            names.append(row[0])
            tmp = cv2.imread('./test/' + row[0], 0)
            th1 = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)
            classes[row[0]] = int(float(row[1]))
            img = th1[int(float(row[4])):int(float(row[5])), int(float(row[2])):int(float(row[3]))]
            res.append(img)
    return res, names, classes

def classification(img1,img2,sift,bf):
    try:
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        points=0
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            for m in matches:
                if m.distance < 224:
                    points += (224-m.distance)
        return points
    except Exception as ex:
        print(ex)

def write(result, names_test,name_file):
    with open(name_file, mode='w') as csvfile:
        spam_writer=csv.writer(csvfile,delimiter=',')
        for i in names_test:
            spam_writer.writerow([i,result[i]])


if __name__=='__main__':
    list_prot,names_prot,classes_prot=read_prot('prot.csv')
    list_test,names_test,classes_test=read_test(sys.argv[1])
    result={i:0 for i in names_test}
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    for i in range(len(list_test)):
        res = list()
        for j in range(len(classes_prot)):
            if len(classes_prot[j]) !=0:
                index=classes_prot[j][0]
                val=classification(list_test[i],list_prot[index],sift,bf)
                res.append(j)
        if len(res)==1:
            index=names_test[i]
            result[index]=res[0]
        elif len(res)==0:
            index = names_test[i]
            result[index]=-1
        else:
            val_max=0
            val_index=0
            for j in res:
                for ind in classes_prot[j]:
                    val = classification(list_test[i], list_prot[ind], sift, bf)
                    if val>val_max:
                        val_index=j
            index = names_test[i]
            result[index]=val_index
    tot=0
    write(result,names_test,sys.argv[2])
    for i in names_test:
        val1=result[i]
        val2=classes_test[i]
        if val1==val2:
            tot+=1
    acc=tot/len(names_test)
    print('Accuracy: '+str(acc))

