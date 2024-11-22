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

from infers.onnx_det import PlateDetector

plate_color_list=['黑色','蓝色','绿色','白色','黄色']
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value,std_value=((0.588,0.193))#识别模型均值标准差
clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

def decodePlate(preds):        #识别后处理
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    plate=""
    for i in newPreds:
        plate+=plateName[int(i)]
    return plate
    # return newPreds

def rec_pre_precessing(img,size=(48,168)): #识别前处理
    img =cv2.resize(img,(168,48))
    img = img.astype(np.float32)
    img = (img/255-mean_value)/std_value  #归一化 减均值 除标准差
    img = img.transpose(2,0,1)         #h,w,c 转为 c,h,w
    img = img.reshape(1,*img.shape)    #channel,height,width转为batch,channel,height,channel
    return img

def get_plate_result(img,session_rec): #识别后处理
    img =rec_pre_precessing(img)
    y_onnx_plate,y_onnx_color = session_rec.run([session_rec.get_outputs()[0].name,session_rec.get_outputs()[1].name], {session_rec.get_inputs()[0].name: img})
    index =np.argmax(y_onnx_plate,axis=-1)
    index_color = np.argmax(y_onnx_color)
    plate_color = plate_color_list[index_color]
    # print(y_onnx[0])
    plate_no = decodePlate(index[0])
    return plate_no,plate_color


def get_split_merge(img):  #双层车牌进行分割后识别
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    new_img = np.hstack((img_upper,img_lower))
    return new_img
    
    
class PlateRecognizer:
    def __init__(self, model_dir):
        self.rec_model = os.path.join(model_dir, 'plate_rec_color.onnx')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        # Initialize models
        self.session_rec = onnxruntime.InferenceSession(self.rec_model, providers=self.providers)
 
    def predict(self, det_results):
        rects,landmarks,labels,scores,roi_imgs = det_results
        for rect, landmark, label, score, roi_img in zip(rects, landmarks, labels, scores, roi_imgs):
            
            if label == 1:  # Dual-layer plate
                roi_img = get_split_merge(roi_img)
            # Get plate results
            img, session_rec = roi_img, self.session_rec
            img =rec_pre_precessing(img)
            y_onnx_plate,y_onnx_color = session_rec.run([session_rec.get_outputs()[0].name,session_rec.get_outputs()[1].name], {session_rec.get_inputs()[0].name: img})
            import pdb;pdb.set_trace()
            index =np.argmax(y_onnx_plate,axis=-1)
            index_color = np.argmax(y_onnx_color)
            plate_color = plate_color_list[index_color]
            plate_no = decodePlate(index[0])

        #     # Save result details
        #     result_dict['rect'] = rect
        #     result_dict['landmarks'] = land_marks.tolist()
        #     result_dict['plate_no'] = plate_no
        #     result_dict['roi_height'] = roi_img.shape[0]
        #     result_dict['plate_color'] = plate_color

        
        # # Return results
        # return [each['plate_no'] for each in dict_list]


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='weights', help='Recognition model path')
    parser.add_argument('--image_path', type=str, default='/data/fuhu.gu/workspace/github/Chinese_license_plate_detection_recognition/imgs/0a39b96d-8278-436a-9e01-543592c4fef2.jpg', help='Image path')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    args = parser.parse_args()
    # Create PlateRecognizer instance
    plate_detector = PlateDetector(args.model_dir, args.img_size)
    # Create PlateRecognizer instance
    plate_recognizer = PlateRecognizer(args.model_dir)
    
    # Perform inference on a single image
    det_result = plate_detector.predict(cv2.imread(args.image_path))
    result_image, plate_texts = plate_recognizer.predict(det_result)

        