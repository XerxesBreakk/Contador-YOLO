import cv2
import datetime

ip_addr = '192.168.1.103'
stream_url = 'http://' + ip_addr + ':81/stream'
videocapture=cv2.VideoCapture(stream_url)

def open_video():
    ip_addr = '192.168.1.103'
    stream_url = 'http://' + ip_addr + ':81/stream'
    videocapture=cv2.VideoCapture(stream_url)
    
    endTime = datetime.datetime.now() + datetime.timedelta(seconds=15)
    video_frame_count=0

    while videocapture.isOpened() and datetime.datetime.now() <= endTime:
        success, frame = videocapture.read()
        if not success:
            break
        video_frame_count+=1
        if video_frame_count == 1:
            cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
        cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    videocapture.release()
    cv2.destroyAllWindows()
    print("end of script")
    return video_frame_count