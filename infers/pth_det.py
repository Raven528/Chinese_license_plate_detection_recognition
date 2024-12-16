# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import time
import copy
import torch
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords 
from plate_recognition.plate_rec import get_plate_result,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge

def four_point_transform(image, pts):                       #透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  #返回到原图坐标
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords

class PlateDetection:
    def __init__(self, model_path, device, img_size):
        self.device = device
        self.img_size = img_size
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """ Load the plate detection model. """
        model = attempt_load(model_path, map_location=self.device)
        return model

    def infer(self, img):
        """ Perform inference to detect plates. """
        img0 = copy.deepcopy(img)
        assert img is not None, 'Image Not Found'
        h0, w0 = img.shape[:2]  # original hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, 0.3, 0.5)  # conf_thres = 0.3, iou_thres = 0.5
        return pred, img.shape[2:], img0.shape

class PlateRecognition:
    def __init__(self, model_path, device, is_color=True):
        self.device = device
        self.is_color = is_color
        self.model = self.init_model(model_path)

    def init_model(self, model_path):
        """ Initialize the plate recognition model. """
        return init_model(self.device, model_path, is_color=self.is_color)
    
    def get_plate_rec_landmark(self, img, xyxy, conf, landmarks, class_num):
        """ Extract the plate and recognize the plate number. """
        result_dict = {}
        x1, y1, x2, y2 = map(int, xyxy)
        landmarks_np = np.zeros((4, 2))
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            landmarks_np[i] = np.array([point_x, point_y])

        class_label = int(class_num)  # 0: single plate, 1: double plate
        roi_img = four_point_transform(img, landmarks_np)  # perspective transform

        if class_label:  # If double plate, split and merge
            roi_img = get_split_merge(roi_img)
        if self.is_color:
            plate_number,rec_prob,plate_color,color_conf = get_plate_result(roi_img, self.device, self.model, is_color=self.is_color)
        else:
            plate_number, rec_prob = get_plate_result(roi_img, self.device, self.model, is_color=self.is_color)

        result_dict['rect'] = [x1, y1, x2, y2]
        result_dict['detect_conf'] = conf
        result_dict['landmarks'] = landmarks_np.tolist()
        result_dict['plate_no'] = plate_number
        result_dict['rec_conf'] = rec_prob
        result_dict['roi_height'] = roi_img.shape[0]

        if self.is_color:
            result_dict['plate_color'] = plate_color
            result_dict['color_conf'] = color_conf

        result_dict['plate_type'] = class_label
        return result_dict

    def infer(self, img, detect_model):
        """ Perform plate recognition inference. """
        dict_list = []
        pred, img_shape, orig_shape = detect_model.infer(img)
        
        # TODO 
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_shape, det[:, :4], orig_shape).round()
                det[:, 5:13] = scale_coords_landmarks(img_shape, det[:, 5:13], orig_shape).round()
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:13].view(-1).tolist()
                    class_num = det[j, 13].cpu().numpy()
                    result_dict = self.get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num)
                    dict_list.append(result_dict)

        return dict_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')
    parser.add_argument('--is_color', type=bool, default=True, help='plate color recognition')
    parser.add_argument('--image_path', type=str, default='imgs/moto.png', help='source image path')
    parser.add_argument('--img_size', type=int, default=640, help='input image size')
    parser.add_argument('--output', type=str, default='result', help='output directory for results')
    opt = parser.parse_args()

    # Create output directory if it doesn't exist
    save_path = opt.output
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize models
    detect_model = PlateDetection(opt.detect_model, device, opt.img_size)
    plate_rec_model = PlateRecognition(opt.rec_model, device, is_color=opt.is_color)

    # Read the input image
    img = cv_imread(opt.image_path)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Perform inference
    start_time = time.time()
    result_dicts = plate_rec_model.infer(img, detect_model)
    end_time = time.time()

    # Output results
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    for res in result_dicts:
        print(res)

if __name__ == '__main__':
    main()

