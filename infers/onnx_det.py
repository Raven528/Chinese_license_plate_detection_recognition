import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]


def order_points(pts):     # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):  #透视变换得到矫正后的图像，方便识别
    rect = order_points(pts)
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

def my_letter_box(img,size=(640,640)):  #
    h,w,c = img.shape
    r = min(size[0]/h,size[1]/w)
    new_h,new_w = int(h*r),int(w*r)
    top = int((size[0]-new_h)/2)
    left = int((size[1]-new_w)/2)
    
    bottom = size[0]-new_h-top
    right = size[1]-new_w-left
    img_resize = cv2.resize(img,(new_w,new_h))
    img = cv2.copyMakeBorder(img_resize,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def xywh2xyxy(boxes):   #xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    xywh =copy.deepcopy(boxes)
    xywh[:,0]=boxes[:,0]-boxes[:,2]/2
    xywh[:,1]=boxes[:,1]-boxes[:,3]/2
    xywh[:,2]=boxes[:,0]+boxes[:,2]/2
    xywh[:,3]=boxes[:,1]+boxes[:,3]/2
    return xywh
 
def my_nms(boxes,iou_thresh):         #nms
    index = np.argsort(boxes[:,4])[::-1]
    keep = []
    while index.size >0:
        i = index[0]
        keep.append(i)
        x1=np.maximum(boxes[i,0],boxes[index[1:],0])
        y1=np.maximum(boxes[i,1],boxes[index[1:],1])
        x2=np.minimum(boxes[i,2],boxes[index[1:],2])
        y2=np.minimum(boxes[i,3],boxes[index[1:],3])
        
        w = np.maximum(0,x2-x1)
        h = np.maximum(0,y2-y1)

        inter_area = w*h
        union_area = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])+(boxes[index[1:],2]-boxes[index[1:],0])*(boxes[index[1:],3]-boxes[index[1:],1])
        iou = inter_area/(union_area-inter_area)
        idx = np.where(iou<=iou_thresh)[0]
        index = index[idx+1]
    return keep

def restore_box(boxes,r,left,top):  #返回原图上面的坐标
    boxes[:,[0,2,5,7,9,11]]-=left
    boxes[:,[1,3,6,8,10,12]]-=top

    boxes[:,[0,2,5,7,9,11]]/=r
    boxes[:,[1,3,6,8,10,12]]/=r
    return boxes



def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  #将识别结果画在图上
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class PlateDetector:
    def __init__(self, model_dir, img_size=640):
        self.detect_model = os.path.join(model_dir, 'plate_detect.onnx')
        self.img_size = img_size
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        
        # Initialize models
        self.session_detect = onnxruntime.InferenceSession(self.detect_model, providers=self.providers)

    def predict(self, image):
        """ Perform inference on a single image. """
        
        # Step 2: Load the image
        try:
            img = image
            if img is None:
                raise ValueError(f"Failed to load image.")
            img0 = copy.deepcopy(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        # Step 3: Preprocess the image for detection
        try:
            img, r, left, top = self.detect_preprocess(img, (self.img_size, self.img_size))
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
        
        # Step 4: Run the detection model
        try:
            y_onnx = self.session_detect.run(
                [self.session_detect.get_outputs()[0].name], 
                {self.session_detect.get_inputs()[0].name: img}
            )[0]
        except Exception as e:
            print(f"Error during detection inference: {e}")
            return None
        
        # Step 5: Post-process detection output
        try:
            rects,landmarks,labels,scores = self.detect_postprocess(y_onnx, r, left, top)
            # get cap-plate-img
            roi_imgs = [four_point_transform(img0, land_marks) for land_marks in landmarks]
        except Exception as e:
            print(f"Error during post-processing: {e}")
            return None
        # Return results
        return (rects,landmarks,labels,scores,roi_imgs)

    def detect_preprocess(self, img, img_size):
        """ Pre-process image for detection. """
        img,r,left,top=my_letter_box(img,img_size)
        img =img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
        img=img/255
        img=img.reshape(1,*img.shape)
        return img,r,left,top

    def detect_postprocess(self,dets,r,left,top,conf_thresh=0.3,iou_thresh=0.5):#检测后处理
        choice = dets[:,:,4]>conf_thresh
        dets=dets[choice]
        dets[:,13:15]*=dets[:,4:5]
        box = dets[:,:4]
        boxes = xywh2xyxy(box)
        score= np.max(dets[:,13:15],axis=-1,keepdims=True)
        index = np.argmax(dets[:,13:15],axis=-1).reshape(-1,1)
        output = np.concatenate((boxes,score,dets[:,5:13],index),axis=1) 
        reserve_=my_nms(output,iou_thresh) 
        output=output[reserve_] 
        output = restore_box(output,r,left,top)
        
        rects = [out[:4].tolist() for out in output]
        scores = [out[4] for out in output]
        land_marks = [out[5:13].reshape(4, 2) for out in output]
        labels = [int(out[13]) for out in output]
        return rects, land_marks, labels, scores

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='weights', help='Detection model path')
    parser.add_argument('--image_path', type=str, default='imgs/moto.png', help='Image path')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    args = parser.parse_args()
    # Create PlateRecognizer instance
    plate_detector = PlateDetector(args.model_dir, args.img_size)
    
    # Perform inference on a single image
    _,_,_,_,scores = plate_detector.predict(cv2.imread(args.image_path))
    print(scores)

