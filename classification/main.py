import sys
import csv
import cv2 as cv
import os
import operator 

TEST_DIR = "test"
INF = 1000000
def classification(args):
    input_file = args[1]
    logos_set = {}
    sift = cv.xfeatures2d.SIFT_create()
    
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        for row in csv_reader:
            if row[1] not in logos_set:
                logos_set[row[1]] = []
            logo = cv.imread(os.path.join("prot", row[0]))
            
            kp, features = sift.detectAndCompute(logo, None)
            if features is not None:
                logos_set[row[1]].append(features)
            
    bf = cv.BFMatcher()
    points_for_class = {}
    results = open(args[2], "w")
    
    for image_path in os.listdir(TEST_DIR):
        image = cv.imread(os.path.join(TEST_DIR, image_path))
        kp, features_image = sift.detectAndCompute(image, None)
        
        for logo_id, array in logos_set.items():
                
            for prot in array:
                if logo_id not in points_for_class:
                    points_for_class[logo_id] = INF
                
                try:
                    matches = bf.match(features_image, prot)
                    
                    matches = sorted(matches, key=lambda x:x.distance)
                    #points_for_class[logo_id] += matches[0].distance
                    
                    for m in matches:
                        if m.distance < 300:
                            if points_for_class[logo_id] == INF:
                                points_for_class[logo_id] = 0
                            points_for_class[logo_id] += m.distance
                except Exception as e :
                    pass
            if logo_id in points_for_class:
                points_for_class[logo_id] = points_for_class[logo_id] / len(array)
        #print(points_for_class)
        id_class = min(points_for_class.items(), key=operator.itemgetter(1))[0]
        results.write(image_path + "," + id_class + "\n")
    results.close()
        
            
if __name__ == "__main__":
    classification(sys.argv)