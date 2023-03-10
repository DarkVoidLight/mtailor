import unittest
from PIL import Image
from model import OnnxModel, Preprocessor


class TestOnnxModel(unittest.TestCase):

    def setUp(self):
        self.model = OnnxModel()
        self.preprocessor = Preprocessor()

    def test_model_output(self):
        test_data = [("./n01440764_tench.jpeg", 0), ("./n01667114_mud_turtle.JPEG", 35)]
        for img_pth, output in test_data:
            img = Image.open(img_pth)
            inp = self.preprocessor.preprocess(img)
            onnx_res = self.model.predict(inp)
            self.assertEqual(onnx_res, output)


if __name__ == '__main__':
    unittest.main()
