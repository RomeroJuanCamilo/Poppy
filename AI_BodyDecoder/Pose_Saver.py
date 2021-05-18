import csv
import os
import numpy as np

def save_funct(results, class_name):

    #Number of coords to train about
    num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
    #print(num_coords)

    
    
    try:
        with open('./data/coords/coords.csv') as f:
            print("File found, adding info")
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row
                
                # Append class name 
                row.insert(0, class_name)
                
                # Export to CSV
                with open('./data/coords/coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 
                
                print("[INFO]: Pose saved!!!!")
            
            except:
                pass
    except IOError:
        print("File created")

        landmarks = ['class']

        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        with open('./data/coords/coords.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)
    
    
    