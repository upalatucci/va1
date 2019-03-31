import sys
import csv
import cv2 as cv
import os
import operator 

TEST_DIR = "test"
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
    count_file = 0
    count_true = 0
    
    with open("gt.csv") as gt_file:
        lines = gt_file.readlines()
        for line in lines:
            file_name, gt = line.split(",")
            image = cv.imread(os.path.join(TEST_DIR, file_name))
            kp, features_image = sift.detectAndCompute(image, None)
            
            for logo_id, array in logos_set.items():
                points_for_class[logo_id] = {}
                count_image_for_logo = 0
                points_for_class[logo_id][count_image_for_logo] = 0
                for prot in array:
                    if count_image_for_logo not in points_for_class[logo_id]:
                        points_for_class[logo_id][count_image_for_logo] = 0
                    count_image_for_logo += 1
                    points = 0
                    try:
                        matches = bf.match(features_image, prot)
                        matches = sorted(matches, key = lambda x:x.distance)
                        
                        for m in matches:
                            if m.distance < 400:
                                points += m.distance
                        
                        #if len(matches) > 0:
                        #    points = points/len(matches)
                        
                    except Exception as e :
                        print(e)
                    
                    points_for_class[logo_id][count_image_for_logo] = points
                
                id_max = max(points_for_class[logo_id].items(), key=operator.itemgetter(1))[0]
                points_for_class[logo_id] = points_for_class[logo_id][id_max]
                
            id_class = max(points_for_class.items(), key=operator.itemgetter(1))[0]
            
            if float(id_class) == float(gt):
                count_true += 1
            count_file += 1
            results.write(file_name + "," + id_class + "\n")
    results.close()
    print(count_true/count_file)
        
            
if __name__ == "__main__":
    classification(sys.argv)