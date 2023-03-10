import torch
from PIL import Image
from pytorch_model import BasicBlock,Classifier


if __name__ == "__main__":

    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    mtailor.eval()

    img = Image.open("./n01667114_mud_turtle.JPEG")
    inp = mtailor.preprocess_numpy(img).unsqueeze(0)
    res = mtailor.forward(inp)

    # Export the model
    torch.onnx.export(mtailor,
                      inp,
                      "mtailor.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True)

    print("Model Successfully Converted to ONNX format")