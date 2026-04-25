import cv2
import math
import cvzone
from ultralytics import YOLO

#loading the trained YOLO model
yolo_model = YOLO("Weights/best.pt")

#defining class names
class_labels = ["Front-Windscreen-Damage", "Headlight-Damage", "Rear-windscreen-Damage",
                "Sidemirror-Damage", "Taillight-Damage", "bonnet-dent",
                "boot-dent", "doorouter-dent", "fender-dent",
                "front-bumper-dent", "quaterpanel-dent", "rear-bumper-dent"]

#loading the input image
image_path = "Media/dent_1.jpg"
img = cv2.imread(image_path)

#object detection using the YOLO model
results = yolo_model(img)

#looping through the detections and drawing bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.4:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

#displaying the output image
cv2.imshow("Image", img)

#closing the windiw when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
