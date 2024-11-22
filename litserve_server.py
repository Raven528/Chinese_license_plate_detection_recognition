from infers.onnx_infer import PlateRecognizer
import litserve as ls
import argparse

# init config
parser = argparse.ArgumentParser()
parser.add_argument('--detect_model', type=str, default='weights/plate_detect.onnx', help='Detection model path')
parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.onnx', help='Recognition model path')
parser.add_argument('--image_path', type=str, default='/path/to/image.jpg', help='Image path')
parser.add_argument('--img_size', type=int, default=640, help='Input image size')
parser.add_argument('--output', type=str, default='./output', help='Output folder path')
args = parser.parse_args()

class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        self.plate_recognizer = PlateRecognizer(args.detect_model, args.rec_model, args.img_size, args.output)

    def decode_request(self, request):
        return request["url"]

    def predict(self, x):
        return self.plate_recognizer.infer_single_image(x)

    def encode_response(self, output):
        car_plates = []
        for plate in output:
            car_plates.append({"plate_no": plate['plate_no'],'plate_color':plate['plate_color']})
        return car_plates

if __name__ == "__main__":
    api = TextClassificationAPI()
    server = ls.LitServer(api, workers_per_device=1)
    server.run(port=8100)
