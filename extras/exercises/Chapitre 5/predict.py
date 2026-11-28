
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import engine, model_builder, utils


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image with a trained TinyVGG")
    parser.add_argument("--hidden_units", type=int, default=10,
                        help="Must match the value used during training")
    parser.add_argument("--image", type=str, default="data/pizza_steak_sushi/test/pizza/194643.jpg",
                        help="Path to the image to classify")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = "data/pizza_steak_sushi/test"

    class_names = sorted(entry.name for entry in Path(test_dir).iterdir() if entry.is_dir())

    # Rebuild the same architecture as during training, then load the saved weights
    loaded_model = model_builder.TinyVGG(input_shape=3,
                                         hidden_units=args.hidden_units,
                                         output_shape=len(class_names))
    loaded_model.load_state_dict(utils.load_model(target_dir="models",
                                                  model_name="05_going_modular_script_mode_tinyvgg_model.pth"))
    loaded_model.to(device)

    # Transform the image exactly like the training images
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = Image.open(args.image).convert("RGB")  # convert() guards against grayscale/RGBA images
    image_tensor = data_transform(image)           # shape: [3, 64, 64]

    # make_predictions expects a list of samples and adds the batch dimension itself
    pred_label = engine.make_predictions(model=loaded_model,
                                         data=[image_tensor],
                                         device=device)

    print(f"[INFO] Image: {args.image}")
    print(f"[INFO] Predicted class: {class_names[pred_label[0]]}")


if __name__ == "__main__":
    main()
