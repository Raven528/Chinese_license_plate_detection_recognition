import os
import cv2
import torch
import argparse
import litserve as ls

from infers.pth_det import PlateDetection, PlateRecognition

# init config
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='weights', help='Recognition model path')
parser.add_argument('--image_path', type=str, default='imgs/moto.png', help='Image path')
parser.add_argument('--img_size', type=int, default=640, help='Input image size')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        # Create PlateRecognizer instance
        self.plate_detector = PlateDetection(os.path.join(args.model_dir, 'det.pth'),device,args.img_size)
        # Create PlateRecognizer instance
        self.plate_recognizer = PlateRecognition(os.path.join(args.model_dir, 'rec.pth'),device,args.model_dir)

    def decode_request(self, request):
        return request["url"]

    def predict(self, x):
        # Perform inference on a single image
        rec_result = self.plate_recognizer.infer(cv2.imread(args.image_path), self.plate_detector)
        return rec_result

    def encode_response(self, output):
        return output

if __name__ == "__main__":
    api = TextClassificationAPI()
    server = ls.LitServer(api)
    server.run(port=8100)
