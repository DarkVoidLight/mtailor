import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# def preprocess_numpy1(img):
#     # convert to RGB if needed
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     resize = transforms.Resize((224, 224))  # must same as here
#     crop = transforms.CenterCrop((224, 224))
#     to_tensor = transforms.ToTensor()
#     # divide by 255
#     img = to_tensor(img) / 255
#     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     img = resize(img)
#     img = crop(img)
#     img = to_tensor(img)
#     img = normalize(img)
#     return img

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



ort_session = onnxruntime.InferenceSession("mtailor.onnx")
to_tensor = transforms.ToTensor()

test_data = [("./n01440764_tench.jpeg", 0), ("./n01667114_mud_turtle.JPEG", 35)]

for img_pth, output in test_data:
    img = Image.open(img_pth)
    inp = preprocess_numpy(img).unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}
    model_output = ort_session.run(None, ort_inputs)[0]
    onnx_res = torch.argmax(to_tensor(model_output).unsqueeze_(0))
    assert onnx_res==output
