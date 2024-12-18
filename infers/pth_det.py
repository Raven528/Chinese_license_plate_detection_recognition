# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import time
import copy
import torch
import argparse
import torchvision
import numpy as np
from torchvision.ops.boxes import box_iou
from ultralytics.utils.ops import xywh2xyxy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

color=['黑色', '蓝色', '绿色', '白色', '黄色']    
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value,std_value=(0.588,0.193)

from ultralytics.utils.checks import check_imgsz
def cv_imread(path):  #可以读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 13  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    multi_label=False
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 14), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 13), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 13] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 13:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 13:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 13, None], x[i, 5:13] ,j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 13:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:13], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 13:14] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

from ultralytics.utils.ops import scale_coords

def decodePlate(preds):
    pre=0
    newPreds=[]
    index=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
            index.append(i)
        pre=preds[i]
    return newPreds,index

def image_processing(img,device):
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img,device,model,is_color=False):
    input = image_processing(img,device)
    if is_color:  #是否识别颜色
        preds,color_preds = model(input)
        color_preds = torch.softmax(color_preds,dim=-1)
        color_conf,color_index = torch.max(color_preds,dim=-1)
        color_conf=color_conf.item()
    else:
        preds = model(input)
    preds=torch.softmax(preds,dim=-1)
    prob,index=preds.max(dim=-1)
    index = index.view(-1).detach().cpu().numpy()
    prob=prob.view(-1).detach().cpu().numpy()

    newPreds,new_index=decodePlate(index)
    prob=prob[new_index]
    plate=""
    for i in newPreds:
        plate+=plateName[i]
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    if is_color:
        return plate,prob,color[color_index],color_conf    #返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate,prob

def get_split_merge(img):
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    new_img = np.hstack((img_upper,img_lower))
    return new_img

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

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
        # model = attempt_load(model_path, map_location=self.device)
        model = torch.load(model_path, map_location=self.device)
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
        imgsz = check_imgsz(self.img_size, stride=self.model.stride.max())  # check img_size
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
        # return init_model(self.device, model_path, is_color=self.is_color)
        model = torch.load(model_path, map_location=self.device)
        return model
    
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
        result_dict['detect_conf'] = conf.item()
        result_dict['landmarks'] = landmarks_np.tolist()
        result_dict['plate_no'] = plate_number
        result_dict['rec_conf'] = rec_prob.tolist()
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
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/det.pth', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default='weights/rec.pth', help='model.pt path(s)')
    parser.add_argument('--is_color', type=bool, default=True, help='plate color recognition')
    parser.add_argument('--image_path', type=str, default='demo.jpg', help='source image path')
    parser.add_argument('--img_size', type=int, default=640, help='input image size')
    opt = parser.parse_args()

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

