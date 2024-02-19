from collections import defaultdict
from shapely.geometry import Polygon
from ultralytics import YOLO
import cv2
import datetime
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from shapely.geometry.point import Point
from ultralytics.utils.files import increment_path
from pathlib import Path

track_history = defaultdict(list)

current_region = None

#Region en la que realizara analisis
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

    #Yolo model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.model.names
    
    # Video setup
    ip_addr = '192.168.1.100'
    stream_url = 'http://' + ip_addr + ':81/stream'
    video="people.mp4"
    videocapture=cv2.VideoCapture(video)
    
    #time to analysis
    endTime = datetime.datetime.now() + datetime.timedelta(seconds=15)

    # Iterate over video frames
    while videocapture.isOpened() and datetime.datetime.now() <= endTime:
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        #Extract the results from the deteccion apply to the frame
        results = model.track(frame, persist=True, classes=classes)

        #Deconstruction of the results from the analysis
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

        # Draw regions (Polygons/Rectangles)
        """ for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness) """
        
        if view_img:
            #if vid_frame_count == 1:
                #cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        #if save_img:
        #    video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    #video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
    print("end of count")

    return final_count


def main():
    """Main function."""
    run()


if __name__ == "__main__":
    main()