from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import datetime

model = YOLO("yolov8n.pt")
in_counts = 0
out_counts = 0
ip_addr = '192.168.1.103'
stream_url = 'http://' + ip_addr + ':81/stream'
cap=cv2.VideoCapture(stream_url) 

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Define region points
# region_points = [(200, 100), (420, 100), (420, 280), (200, 280)]
region_points = [(300, 100), (300, 400)]

'''
# Video writer
video_writer = cv2.VideoWriter("/Users/saulmestanza/Documents/workspace/doraemon/object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
'''

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=True,
    reg_pts=region_points,
    classes_names={0: 'person'},
    draw_tracks=True,
)
endTime = datetime.datetime.now() + datetime.timedelta(seconds=30)

print("INICIO!")

while cap.isOpened() and datetime.datetime.now() <= endTime:
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True, verbose=False, classes=[0])

    try:
        # im0 = counter.start_counting(im0, tracks)
        counter.start_counting(im0, tracks)
        in_counts = counter.in_counts
        out_counts = counter.out_counts
        print("counter.in_counts: ", in_counts)
        print("counter.out_counts: ", out_counts)
    except Exception as e:
        print("************", e)
    # video_writer.write(im0)

print("TERMINA!")
print(in_counts, out_counts)
cap.release()
# video_writer.release()
cv2.destroyAllWindows()