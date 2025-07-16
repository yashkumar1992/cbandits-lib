#!/usr/bin/env python3
import argparse
import os
from open_bandits_adapter.adaptor import OpenBanditsAdapter
from core.utils.data_loader import load_dat  # implement this to parse your .dat files

def main():
    parser = argparse.ArgumentParser("Train and Demo Open Bandits Model")
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(os.path.dirname(__file__), './config/config.yaml'),
        help='Path to adapter config file'
    )
    parser.add_argument(
        '--train-data', type=str, required=True,
        help='.dat file with training interactions'
    )
    parser.add_argument(
        '--output', type=str, default='ob_model',
        help='Output model name'
    )
    parser.add_argument(
        '--demo-data', type=str,
        help='.dat file to run demo predictions'
    )
    args = parser.parse_args()

    # Initialize adapter
    adapter = OpenBanditsAdapter(args.config)

    # Training phase
    adapter.batch_update(data_file=args.train_data)
    version = adapter.save(name=args.output, version='v1', registry_root='./registry')
    print(f"Model saved at version: {version}")

    # Optional demo predictions
    if args.demo_data:
        print("\nRunning demo predictions:")
        rows = load_dat(args.demo_data)
        for context, actions in rows:
            action_id, score = adapter.predict((context, actions))
            print(f"Chosen: {action_id} (score={score:.4f})")

if __name__ == '__main__':
    main()