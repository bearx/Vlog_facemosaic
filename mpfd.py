import dlib
import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

class mpfd(object):
    def __init__(self):
        self.detector=mp.solutions.face_detection.FaceDetection()
        #self.drawing=mp.solutions.drawing_utils
    def __call__(self,frame):
        frame_t=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=self.detector.process(frame_t)
        resultset=[]
        if result.detections:
            image_rows,image_cols,_=frame.shape
            for detect in result.detections:
                #self.drawing.draw_detection(frame,detect)
                if not detect.location_data:
                    continue
                location=detect.location_data
                relative_bounding_box=location.relative_bounding_box
                rect_start_point=_normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin,relative_bounding_box.ymin,image_cols,
                    image_rows)
                rect_end_point=_normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin+relative_bounding_box.width,
                    relative_bounding_box.ymin+relative_bounding_box.height,image_cols,
                    image_rows)
                try:
                    rs=dlib.rectangle(rect_start_point[0],rect_start_point[1],rect_end_point[0],rect_end_point[1])
                except:
                    continue
                resultset.append(rs)
        return resultset
if __name__=="__main__":
    r=mpfd()