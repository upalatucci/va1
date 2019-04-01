import sys
import csv
import cv2 as cv
import os
import operator 

TEST_DIR = "test"
THRESHOLD = 400

def build_prot_dict(sift):
    logos_set = {}
    with open("prot.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        for row in csv_reader:
            if row[1] not in logos_set:
                logos_set[row[1]] = []
            logo = cv.imread(os.path.join("prot", row[0]))
            
            kp, features = sift.detectAndCompute(logo, None)
            if features is not None:
                logos_set[row[1]].append(features)
    return logos_set
    
def classification(args):
    input_file = args[1]
    logos_set = {}
    sift = cv.xfeatures2d.SIFT_create()
    logos_set = build_prot_dict(sift)
    bf = cv.BFMatcher()
    points_for_class = {}
    results = open(args[2], "w")
    
    for file_name in os.listidr(input_file):
        id_class = extract_classification(sift, bf, file_name, logos_set)
        results.write(file_name + "," + id_class + "\n")
    results.close()
    

def classification_with_accuracy(args):
    input_file = args[1]
    logos_set = {}
    sift = cv.xfeatures2d.SIFT_create()
    logos_set = build_prot_dict(sift)
    
    bf = cv.BFMatcher()
    results = open(args[2], "w")
    count_file = 0
    count_true = 0
    
    with open(input_file) as gt_file:
        lines = gt_file.readlines()
        for file_name in lines:
            
            id_class = extract_classification(sift, bf, file_name, logos_set)
            
            #if float(id_class) == float(gt):
            #    count_true += 1
            #count_file += 1
            results.write(file_name + "," + id_class + "\n")
    results.close()
    #print(count_true/count_file)
    

def extract_classification(sift, bf, file_name, logos_set):
    points_for_class = {}
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
                    if m.distance < THRESHOLD:
                        points += (THRESHOLD - m.distance)
                
                #if len(matches) > 0:
                #    points = points/len(matches)
                
            except Exception as e :
                print(e)
            
            points_for_class[logo_id][count_image_for_logo] = points
        
        id_max = max(points_for_class[logo_id].items(), key=operator.itemgetter(1))[0]
        points_for_class[logo_id] = points_for_class[logo_id][id_max]
        
    return max(points_for_class.items(), key=operator.itemgetter(1))[0]
        
            
if __name__ == "__main__":
    classification_with_accuracy(sys.argv)