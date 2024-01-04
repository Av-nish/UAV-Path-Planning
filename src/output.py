import cv2
from ultralytics import YOLO


class Output:

    def __init__(self, video_path, model_path, output_path) -> None:


        self.model = YOLO(model_path)
        self.vdo = cv2.VideoCapture(video_path)
        width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width, self.height = self.resize_wh(width, height)
        fps = self.vdo.get(cv2.CAP_PROP_FPS)
        # fps = 1
        self.out = cv2.VideoWriter(filename=output_path, fourcc=cv2.VideoWriter_fourcc(
            *'mp4v'), fps=fps, frameSize=(width, height))

    def resize_wh(self, orig_w, orig_h):

        imgsz = 1280  # image size of yolo default is 640
        new_h = imgsz/orig_w * orig_h  # new height with same ratio
        w = imgsz
        remainder = (new_h % 32)  # YOLO need w,h that can divide by 32
        if remainder > 32/2:
            h = int(new_h - remainder + 32)
        else:
            h = int(new_h - remainder)
        return (w, h)

    # def get_obstacle_coordinates(self, results, frame, current_coors):
    
    #     box = results[0].boxes.xywh

    #     # obstacle_radius = 50

    #     # for coor in box:
    #     #     cv2.circle(frame, (int(coor[0]), int(
    #     #         coor[1])), obstacle_radius, (0, 0, 255), thickness=3)
    #     #     cv2.circle(frame, (int(coor[0]), int(
    #     #         coor[1])), obstacle_radius+50, (255, 0, 0), thickness=2)
            
    #     # for i in range(len(path)-1):
    #         # cv2.line(final_img, (path[i][1], path[i][0]),
    #         #          (path[i+1][1], path[i+1][0]), (0, 255, 0), thickness=10)
            
    #     for uav in current_coors:
    #         final_img = cv2.circle(frame, (uav.x, uav.y),
    #                             20, color=(255, 16, 240), thickness=-1)
        
    #     # cv2.imshow('Video', final_img)
    #     return box, final_img 

        
        
