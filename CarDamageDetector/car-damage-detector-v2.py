import cv2
import math
import cvzone
from ultralytics import YOLO

yolo_model = YOLO(r"CarDamageDetector\Weights\best.pt")

class_labels = ["Front-Windscreen-Damage", "Headlight-Damage", "Rear-windscreen-Damage",
                "Sidemirror-Damage", "Taillight-Damage", "bonnet-dent",
                "boot-dent", "doorouter-dent", "fender-dent",
                "front-bumper-dent", "quaterpanel-dent", "rear-bumper-dent"]

CONF_HIGH = 0.4
CONF_LOW  = 0.2

image_path = r"CarDamageDetector\Media\dent_2.jpg"
img = cv2.imread(image_path)

results = yolo_model(img)

high_conf_detections = []
low_conf_detections  = []

for r in results:
    for box in r.boxes:
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls  = int(box.cls[0])
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        w, h = x2 - x1, y2 - y1

        if conf >= CONF_HIGH:
            high_conf_detections.append((x1, y1, w, h, conf, cls))
        elif conf >= CONF_LOW:
            low_conf_detections.append((x1, y1, w, h, conf))

if not high_conf_detections and not low_conf_detections:
    cvzone.putTextRect(img, "No Damage Detected", (50, 50), scale=1.5, thickness=2, colorR=(0, 200, 0))
else:
    for (x1, y1, w, h, conf, cls) in high_conf_detections:
        cvzone.cornerRect(img, (x1, y1, w, h), t=2)
        cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

    for (x1, y1, w, h, conf) in low_conf_detections:
        cvzone.cornerRect(img, (x1, y1, w, h), t=2, colorC=(0, 100, 255))
        cvzone.putTextRect(img, f'Unknown Damage {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(0, 100, 255))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
