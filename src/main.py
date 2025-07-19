import numpy as np
import cv2 as cv

# Paths to the model files
prototxt = r'C:\Users\firoz\PycharmProjects\real_time_object_detection\model_files\MobileNetSSD_deploy.prototxt'
model = r'C:\Users\firoz\PycharmProjects\real_time_object_detection\model_files\MobileNetSSD_deploy.caffemodel'

# Load the model
net = cv.dnn.readNetFromCaffe(prototxt, model)

# Class labels MobileNetSSD is trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open webcam or video
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input blob
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    # Loop over detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                       frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow("SSD Object Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

