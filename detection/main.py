import cv2
import csv

def read(file_name):
    res=list()
    names=list()
    with open(file_name) as csvfile:
        spam_reader=csv.reader(csvfile, delimiter=',')
        for row in spam_reader:
            names.append(row[0])
            if 'prot' in file_name:
                img= cv2.imread('./prot/'+row[0])
            else:
                img=cv2.imread('./test/'+row[0])
            res.append(img)
    return res,names

def classification(img1,img2,sift,bf):
    try:
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        points=0
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            for m in matches:
                if m.distance < 280:
                    points += m.distance
        return points
    except Exception as ex:
        print(ex)

def write(result, names_test):
    with open('result.csv', mode='w') as csvfile:
        spam_writer=csv.writer(csvfile,delimiter=',')
        for i in names_test:
            tmp=[i]
            row=tmp.extend(result[i])
            spam_writer.writerow(row)


if __name__=='__main__':
    list_prot,names_prot=read('prot.csv')
    list_test,names_test=read('gt.csv')
    result={i:list() for i in names_test}
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    print('Inizio classificazione')
    for i in range(len(list_test)):
        print('Immagine '+names_test[i])
        for j in range(len(list_prot)):
            val=classification(list_test[i],list_prot[j],sift,bf)
            if val>2000:
                res=[j,val]
                index=names_test[i]
                result[index].append(res)
    print('Fine classificazione')
    write(result,names_test)