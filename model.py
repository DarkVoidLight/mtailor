import torch
import onnxruntime
from torchvision import transforms


class OnnxModel:
    def __int__(self):
        self.model = onnxruntime.InferenceSession("mtailor.onnx")
        self.to_tensor = transforms.ToTensor()

    def predict(self, inp):
        ort_inputs = {self.model.get_inputs()[0].name: self.to_numpy(inp)}
        model_output = self.model.run(None, ort_inputs)[0]
        onnx_res = torch.argmax(self.to_tensor(model_output).unsqueeze_(0))
        return onnx_res

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, img):
        self.preprocess_numpy(img).unsqueeze(0)

    def preprocess_numpy(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        resize = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        img = resize(img)
        img = to_tensor(img)
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        img = (img - mean) / std
        return img

