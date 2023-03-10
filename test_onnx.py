import unittest
import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess_numpy(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    resize = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    to_tensor = transforms.ToTensor()
    img = resize(img)
    img = to_tensor(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    img = (img - mean) / std
    return img


class TestOnnxModel(unittest.TestCase):

    def setUp(self):
        self.ort_session = onnxruntime.InferenceSession("mtailor.onnx")
        self.to_tensor = transforms.ToTensor()

    def test_model_output(self):
        test_data = [("./n01440764_tench.jpeg", 0), ("./n01667114_mud_turtle.JPEG", 35)]
        for img_pth, output in test_data:
            img = Image.open(img_pth)
            inp = preprocess_numpy(img).unsqueeze(0)
            ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(inp)}
            model_output = self.ort_session.run(None, ort_inputs)[0]
            onnx_res = torch.argmax(self.to_tensor(model_output).unsqueeze_(0))
            self.assertEqual(onnx_res, output)


if __name__ == '__main__':
    unittest.main()
