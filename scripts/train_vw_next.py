#!/usr/bin/env python3
# scripts/train_vw_next.py

import argparse
from adapters.vw.vw_next_adapter import VWNextCBModel

def main():
    parser = argparse.ArgumentParser(description="Train VW-Next contextual bandit model using ADF format.")
    parser.add_argument('--data-file', '-d',default='datasets/vw_bandit_dataset.dat', help='Path to VW-format dataset file')
    parser.add_argument('--model-name', '-n', default='vw_model', help='Model name for saving')
    parser.add_argument('--model-version', '-v', default='v1', help='Model version for saving')
    parser.add_argument('--registry-root', '-r', default='artifacts', help='Registry root path to save model')
    parser.add_argument('--vw-args', nargs=argparse.REMAINDER, required=True, help='VW arguments (e.g. --cb_adf)')
    args = parser.parse_args()

    model = VWNextCBModel(vw_args=args.vw_args)

    # Train from file using VW's TextFormatReader under the hood
    model.batch_update(data_file=args.data_file)

    # Save the trained model
    path = model.save(args.model_name, args.model_version, args.registry_root)
    print(f"Model saved at: {path}")

if __name__ == '__main__':
    main()
