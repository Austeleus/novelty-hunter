#!/usr/bin/env python3
"""Generate predictions for test set."""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.dataset import NoveltyHunterTestDataset
from src.data.transforms import get_val_transforms
from src.models.model import create_model
from src.ood.detector import create_ood_detector
from src.calibration.temperature_scaling import TemperatureScaling
from src.inference.predictor import Predictor, SimplePredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--ood-detector',
        type=str,
        default='outputs/ood_detector.pt',
        help='Path to OOD detector state'
    )
    parser.add_argument(
        '--calibrator',
        type=str,
        default='outputs/calibrator.pt',
        help='Path to calibrator state (optional)'
    )
    parser.add_argument(
        '--no-ood',
        action='store_true',
        help='Disable OOD detection (simple argmax predictions)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions/test_predictions.csv',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--with-confidence',
        action='store_true',
        help='Include confidence scores in output'
    )
    parser.add_argument(
        '--with-probabilities',
        action='store_true',
        help='Include calibrated probability distributions in output (for cross-entropy eval)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")

    device = args.device
    print(f"Using device: {device}")

    # Create test dataset
    print("\nCreating test dataset...")
    val_transform = get_val_transforms(cfg)

    test_dataset = NoveltyHunterTestDataset(
        img_dir=cfg.data.test_images,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Test samples: {len(test_dataset)}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully")

    if args.no_ood:
        # Simple predictions without OOD detection
        print("\nGenerating predictions (without OOD detection)...")
        predictor = SimplePredictor(model, device=device)
        df = predictor.predict(test_loader, args.output)

    else:
        # Load OOD detector
        print(f"\nLoading OOD detector from {args.ood_detector}...")
        ood_detector = create_ood_detector(cfg, device=device)

        if os.path.exists(args.ood_detector):
            ood_state = torch.load(args.ood_detector, map_location=device)
            ood_detector.load_state_dict(ood_state)
            print("  OOD detector loaded successfully")
        else:
            print(f"  ERROR: OOD detector file not found at {args.ood_detector}")
            print("  Please run tune_thresholds.py first to create the OOD detector,")
            print("  or use --no-ood flag for simple predictions without OOD detection.")
            sys.exit(1)

        # Load calibrator (optional)
        calibrator = None
        if os.path.exists(args.calibrator):
            print(f"\nLoading calibrator from {args.calibrator}...")
            calibrator = TemperatureScaling()
            cal_state = torch.load(args.calibrator, map_location=device)
            calibrator.load_state_dict(cal_state)
            print("  Calibrator loaded successfully")

        # Create predictor
        predictor = Predictor(
            model=model,
            ood_detector=ood_detector,
            calibrator=calibrator,
            cfg=cfg,
            device=device
        )

        # Generate predictions
        print("\nGenerating predictions...")
        if args.with_probabilities:
            df = predictor.predict_with_probabilities(test_loader, args.output)
        elif args.with_confidence:
            df = predictor.predict_with_confidence(test_loader, args.output)
        else:
            df = predictor.predict(test_loader, args.output)

    # Print summary
    print("\n" + "="*50)
    print("Prediction Summary")
    print("="*50)
    print(f"Total predictions: {len(df)}")

    # Superclass distribution
    print("\nSuperclass distribution:")
    super_counts = df['superclass_index'].value_counts().sort_index()
    superclass_names = ['bird', 'dog', 'reptile', 'novel']
    for idx, count in super_counts.items():
        name = superclass_names[idx] if idx < len(superclass_names) else f'class_{idx}'
        print(f"  {name}: {count} ({100*count/len(df):.1f}%)")

    print(f"\nPredictions saved to: {args.output}")


if __name__ == '__main__':
    main()
