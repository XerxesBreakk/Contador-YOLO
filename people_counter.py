from collections import defaultdict
from shapely.geometry import Polygon
from ultralytics import YOLO
import cv2
import datetime
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from shapely.geometry.point import Point

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(10, 10), (500, 10), (500, 550), (10, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=True,
    save_img=False,
    exist_ok=False,
    classes=[0],
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    vid_frame_count = 0
    final_count=0

    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.model.names
    
    # Video setup
    ip_addr = '192.168.1.101'
    stream_url = 'http://' + ip_addr + ':81/stream'
    videocapture=cv2.VideoCapture(stream_url)

    endTime = datetime.datetime.now() + datetime.timedelta(seconds=15)

    # Iterate over video frames
    while videocapture.isOpened() and datetime.datetime.now() <= endTime:
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1
            if region['counts']>final_count:
                final_count=region['counts']
        
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                #cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    #video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

    return final_count