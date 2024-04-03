# Import the Libraries
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model=YOLO('yolov9c.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
              'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
              'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker=CustomTracker()
count=0

# Replace the video with your desired video
cap = cv2.VideoCapture('traffictrim.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the modified frames
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' based on your system
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Looping over each frame and Performing the Detection

down = {}
counter_down = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    results = model.predict(frame)

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    # print(px)

    list = []

    for index, row in px.iterrows():
        #        print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    # print(bbox_id)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        # cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        # cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        y = 308
        offset = 7

        ''' condition for red line '''
        if y < (cy + offset) and y > (cy - offset):
            ''' this if condition is putting the id and the circle on the object when the center of the object touched the red line.'''

            down[id] = cy  # cy is current position. saving the ids of the cars which are touching the red line first.
            # This will tell us the travelling direction of the car.
            if id in down:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                #cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                counter_down.add(id)

                # # line
    text_color = (255, 255, 255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)

    # print(down)
    cv2.line(frame, (282, 308), (1004, 308), red_color, 3)  # starting cordinates and end of line cordinates
    cv2.putText(frame, ('red line'), (280, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


    downwards = (len(counter_down))
    cv2.putText(frame, ('Vehicle Counter - ') + str(downwards), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1,
                cv2.LINE_AA)

    cv2.line(frame,(282,308),(1004,308),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('red line'),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    # This will write the Output Video to the location specified above
    out.write(frame)