#!/usr/bin/env python3
"""
Inference script for satellite image classification.
Processes all .tif files from input directory and writes predictions to CSV.
"""

import os
import sys
import glob
import argparse
import csv
from pathlib import Path
import torch
from transformers import ViTForImageClassification
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


# Class mapping based on filename prefixes
CLASS_MAPPING = {
    'smoke': 0,
    'wildfire': 0,  # wildfire is also smoke
    'haze': 1,
    'cloud': 2,
    'land': 2,
    'seaside': 2,
    'dust': 2,
    'normal': 2
}


def get_actual_class_from_filename(filename):
    """Extract actual class from filename prefix."""
    basename = os.path.basename(filename)
    prefix = basename.split('_')[0].lower()

    # Handle special cases
    if prefix == 'wildfire':
        return 0  # smoke

    return CLASS_MAPPING.get(prefix, 2)  # default to normal (2)


def get_transforms(image_size=384):
    """Get validation transforms for inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, device):
    """Load the trained model."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract config if available
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'google/vit-base-patch16-384')
    num_labels = 3  # smoke, haze, normal

    print(f"Model architecture: {model_name}")

    # Load model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model


def run_inference(model, test_dir, output_csv, device, image_size=384):
    """Run inference on all test images and save predictions to CSV."""

    # Get all .tif files
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.tif')))

    if not test_files:
        print(f"Error: No .tif files found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(test_files)} test images")

    # Get transforms
    transform = get_transforms(image_size)

    # Prepare output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'predicted_class', 'actual_class'])

        # Process each image
        print("Running inference...")
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Processing images"):
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    pixel_values = transform(image).unsqueeze(0).to(device)

                    # Run inference
                    outputs = model(pixel_values=pixel_values)
                    predicted_class = torch.argmax(outputs.logits, dim=1).item()

                    # Get actual class from filename
                    filename = os.path.basename(img_path)
                    actual_class = get_actual_class_from_filename(filename)

                    # Write to CSV
                    writer.writerow([filename, predicted_class, actual_class])

                    # Flush after each write to ensure real-time updates
                    csvfile.flush()

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    print(f"Predictions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on satellite images'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/app/weights/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='/data/test',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/data/output/predictions.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=384,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found at {args.test_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.model_path, device)

    # Run inference
    run_inference(
        model=model,
        test_dir=args.test_dir,
        output_csv=args.output,
        device=device,
        image_size=args.image_size
    )

    print("Inference completed successfully!")


if __name__ == '__main__':
    main()
