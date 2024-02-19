import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np


LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)

def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    model= YOLO("yolov8n.pt")
    cap=cv2.VideoCapture("people.mp4")
    #print("pass this part")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()