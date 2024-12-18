import cv2
import torch
import argparse
import litserve as ls

from infers.detect_plate import detect_Recognition_plate, load_model
from plate_recognition.plate_rec import init_model

# init config
parser = argparse.ArgumentParser()
parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt', help='model.pt path(s)')  #检测模型
parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')#车牌识别+颜色识别模型
parser.add_argument('--image_path', type=str, default='demo.jpg', help='')
parser.add_argument('--img_size', type=int, default=640, help='Input image size')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        self.detect_model = load_model(args.detect_model, device)  #初始化检测模型
        self.plate_rec_model=init_model(device, args.rec_model, is_color=True)      #初始化识别模型
        self.device = device

    def decode_request(self, request):
        self.img = cv2.imread(args.image_path)
        return request["url"]

    def predict(self, x):
        dict_list=detect_Recognition_plate(self.detect_model, self.img, self.device,\
                                           self.plate_rec_model, args.img_size, is_color=True)
        result = max(dict_list, key=lambda x: x['detect_conf']) if dict_list else {}
        return result

    def encode_response(self, output):
        return output

if __name__ == "__main__":
    api = TextClassificationAPI()
    server = ls.LitServer(api)
    server.run(port=8100)
