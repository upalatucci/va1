import sys
import csv
import cv2 as cv


def classification(args):
    input_file = args[1]
    logos_set = {}
    
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        for row in csv_reader:
            if row[1] not in logos_set:
                logos_set[row[1]] = []
            
            
    
    

if __name__ == "__main__":
    classification(sys.argv)