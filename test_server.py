import argparse
from time import perf_counter
import banana_dev as banana
from PIL import Image
from model import OnnxModel, Preprocessor

model = OnnxModel()
preprocessor = Preprocessor()
api_key = "YOUR_API_KEY_HERE"
model_key = "YOUR_MODEL_KEY"

def predict_image(image_path):

    img = Image.open(image_path)
    inp = preprocessor.preprocess(img)
    model_inputs = {model.ort_session.get_inputs()[0].name: model.to_numpy(inp)}

    # Call the deployed model on the Banana Dev platform
    start_time = perf_counter()
    out = banana.run(api_key, model_key, model_inputs)
    print("Time for api call in seconds:", perf_counter() - start_time)

    return out["modelOutputs"][0]['output']


# Define the function to run the tests
def run_tests():

    test_cases = [("path/to/image1.jpg", "class1"), ("path/to/image2.jpg", "class2"), ...]

    # Define the number of correct predictions
    num_correct = 0
    for image_path, expected_class in test_cases:
        pred_class = predict_image(image_path)
        if pred_class == expected_class:
            num_correct += 1
    print("%d/%d tests passed" % (num_correct, len(test_cases)))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="path to the image to predict")
    parser.add_argument("--test", action="store_true", help="run preset custom tests")
    args = parser.parse_args()

    pred_class = predict_image(args.image_path)
    print("Predicted class:", pred_class)

    # If the test flag is set, run the tests
    if args.test:
        run_tests()


# Call the main function
if __name__ == "__main__":
    main()
