#!/usr/bin/env python3
# scripts/train_vw_next.py

import os, sys
#  ───────────────────────────────────────────────────────────────────────────
#  Make sure the parent folder (code/bandit-lib) is on sys.path so that
#  `import adapters.vw.vw_next_adapter` can resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#

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

    # 1) Load back via the adapter
    model.load(
        name=args.model_name,
        version=args.model_version,
        registry_root=args.registry_root,
        vw_args=["--cb_explore_adf","--epsilon=0.5"]  # or any other args you want to pass
        )
    print("🔄 Model reloaded from adapter.load(...)")

    # Quick smoke‐test on your 4th ADF block (dayofweek=3)
    test_context = {"dayofweek": 3}
    test_actions = [
        {"eventType": "views",     "timeWindow": 1440, "threshold": "Minimum: 10"},
        {"eventType": "atc",       "timeWindow": 480,  "threshold": "10-20"},
        {"eventType": "purchases", "timeWindow": 60,   "threshold": "20-30"},
    ]


    # 3) Run an ADF‐style predict
    action_index, p_logged = model.predict((test_context, test_actions))
    print(f" Chosen action: {action_index}   p_logged={p_logged:.4f}")


if __name__ == '__main__':
    main()
