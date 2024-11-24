from infers.onnx_det import PlateDetector
from infers.onnx_rec import PlateRecognizer
import litserve as ls
import argparse
import cv2

# init config
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='weights', help='Recognition model path')
parser.add_argument('--image_path', type=str, default='imgs/moto.png', help='Image path')
parser.add_argument('--img_size', type=int, default=640, help='Input image size')
args = parser.parse_args()

class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        # Create PlateRecognizer instance
        self.plate_detector = PlateDetector(args.model_dir, args.img_size)
        # Create PlateRecognizer instance
        self.plate_recognizer = PlateRecognizer(args.model_dir)

    def decode_request(self, request):
        return request["url"]

    def predict(self, x):
        # Perform inference on a single image
        det_result = self.plate_detector.predict(cv2.imread(args.image_path))
        rec_result = self.plate_recognizer.predict(det_result)
        return rec_result

    def encode_response(self, output):
        return output

if __name__ == "__main__":
    api = TextClassificationAPI()
    server = ls.LitServer(api, workers_per_device=1)
    server.run(port=8100)
