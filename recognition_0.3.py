import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet('robot_yolo\yolo_w\yolov3.cfg', 'robot_yolo\yolo_w\yolov3.weights')

classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck',
'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard','sports ball', 'kite', 'baseball bat', 'baseball glove',
'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant',
'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

living = ['plant', 'person', 'mouse', 'cat', 'dog', 'bird'] 

nonliving = ['chair', 'table', 'monitor', 'keyboard',  
                      'mouse', 'laptop', 'book', 'bag', 'phone',
                      'bottle', 'cup', 'clock', 'whiteboard', 'projector',  
                      'tv', 'desktop', 'printer', 'poster', 'door', 'window',
                      'notebook', 'pen', 'speaker', 'board', 'professor',
                      'student']

# Адрес видоса
video = 'robot_yolo/rgb_output.avi'
cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    boxes = []
    scores = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence > 0.5:
                cx, cy, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                scores.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
    
    for i in indices:
        box = boxes[i]
        class_id = class_ids[i]
        class_name = classes[class_id]
        if class_name in living:
            label = 'living'
        elif class_name in nonliving:
            label = 'nonliving'

        x, y, w, h = box
        x1 = x
        x2 = x + w
        
        print(x1, x2, label)

        # Отрисовка для наглядности
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()