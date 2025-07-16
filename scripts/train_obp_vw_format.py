#!/usr/bin/env python3
# scripts/train_obp_vw_format.py

import os
import sys
import yaml
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.opb.opb_adapter_vw import OpenBanditsAdapterVW, SimpleActionVW
from core.utils.vw_parser import VWFormatParser

def create_vw_config(algorithm='linear_ucb'):
    """Create configuration for different OBP algorithms"""
    configs = {
        'linear_ucb': {
            'open_bandits_adapter': {
                'algorithm': 'linear_ucb',
                'params': {
                    'random_state': 12345
                }
            }
        },
        'logistic_ucb': {
            'open_bandits_adapter': {
                'algorithm': 'logistic_ucb', 
                'params': {
                    'random_state': 12345
                }
            }
        },
        'linear_ts': {
            'open_bandits_adapter': {
                'algorithm': 'linear_ts',
                'params': {
                    'random_state': 12345
                }
            }
        },
        'epsilon_greedy': {
            'open_bandits_adapter': {
                'algorithm': 'epsilon_greedy',
                'params': {
                    'epsilon': 0.1,
                    'random_state': 12345
                }
            }
        },
        'linear_epsilon_greedy': {
            'open_bandits_adapter': {
                'algorithm': 'linear_epsilon_greedy',
                'params': {
                    'epsilon': 0.1,
                    'random_state': 12345
                }
            }
        },
        'random': {
            'open_bandits_adapter': {
                'algorithm': 'random',
                'params': {
                    'random_state': 12345
                }
            }
        }
    }
    
    config = configs.get(algorithm, configs['linear_ucb'])
    
    os.makedirs('conf', exist_ok=True)
    config_path = f'conf/obp_{algorithm}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_sample_vw_data():
    """Create sample VW format data matching your existing file structure"""
    os.makedirs('datasets', exist_ok=True)
    
    sample_vw_data = """shared |C dayofweek=1
|A eventType=views timeWindow=1440 threshold=Minimum: 10
1:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=2
0:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=3
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=atc timeWindow=480 threshold=10-20
2:0.8:0.33 |A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=4
1:0.9:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=5
|A eventType=views timeWindow=1440 threshold=Minimum: 10
0:0.7:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=6
2:1.2:0.33 |A eventType=purchases timeWindow=60 threshold=20-30
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=atc timeWindow=480 threshold=10-20

shared |C dayofweek=7
|A eventType=views timeWindow=1440 threshold=Minimum: 10
2:0.6:0.33 |A eventType=purchases timeWindow=60 threshold=20-30
|A eventType=atc timeWindow=480 threshold=10-20"""
    
    data_path = 'datasets/vw_bandit_dataset.dat'
    with open(data_path, 'w') as f:
        f.write(sample_vw_data)
    
    return data_path

def test_algorithm(algorithm_name: str, data_path: str):
    """Test a specific algorithm"""
    print(f"\n🧪 Testing {algorithm_name.upper()}")
    print("=" * 50)
    
    # Create config for this algorithm
    config_path = create_vw_config(algorithm_name)
    print(f"✅ Created config: {config_path}")
    
    # Initialize adapter
    print(f"🔧 Initializing {algorithm_name} adapter...")
    adapter = OpenBanditsAdapterVW(config_path)
    
    # Train
    print(f"🎯 Training {algorithm_name} model...")
    adapter.batch_update(data_file=data_path)
    
    # Save model
    print("💾 Saving model...")
    os.makedirs('artifacts', exist_ok=True)
    save_path = adapter.save(name=f'obp_{algorithm_name}', version='v1', registry_root='artifacts')
    print(f"✅ Model saved at: {save_path}")
    
    # Test predictions
    print("🔮 Testing predictions...")
    
    # Parse test data for predictions
    parser = VWFormatParser()
    test_interactions = parser.parse_vw_file(data_path)
    
    # Test on first few examples
    for i, (context, actions, label_info) in enumerate(test_interactions[:3]):
        action_objects = [SimpleActionVW(j, action) for j, action in enumerate(actions)]
        
        predicted_action, prob = adapter.predict((context, action_objects))
        
        print(f"\n  Test {i+1}:")
        print(f"    Context: {context}")
        print(f"    Actions: {[f'Action {j}: {action}' for j, action in enumerate(actions)]}")
        print(f"    Predicted action: {predicted_action} (prob={prob:.3f})")
        if label_info:
            true_action, reward, true_prob = label_info
            print(f"    True action: {true_action} (reward={reward:.3f})")
    
    return adapter

def main():
    parser = argparse.ArgumentParser(description="Train OBP models on VW format data")
    parser.add_argument('--algorithm', '-a', default='all', 
                       choices=['all', 'linear_ucb', 'logistic_ucb', 'linear_ts', 
                               'epsilon_greedy', 'linear_epsilon_greedy', 'random'],
                       help='Algorithm to train (default: all)')
    parser.add_argument('--data', '-d', default='datasets/vw_bandit_dataset.dat',
                       help='Path to VW format data file')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate sample data if file does not exist')
    
    args = parser.parse_args()
    
    print("🚀 Starting OBP VW Format Training Demo")
    print("=" * 60)
    
    # Handle data file
    if os.path.exists(args.data):
        data_path = args.data
        print(f"📂 Using existing data: {data_path}")
    elif args.generate_data:
        data_path = create_sample_vw_data()
        print(f"✅ Created sample data: {data_path}")
    else:
        print(f"❌ Data file {args.data} not found. Use --generate-data to create sample data.")
        return
    
    # Parse and display data info
    parser = VWFormatParser()
    interactions = parser.parse_vw_file(data_path)
    print(f"📊 Loaded {len(interactions)} interactions from VW file")
    
    # Display sample interaction
    if interactions:
        context, actions, label_info = interactions[0]
        print(f"📄 Sample interaction:")
        print(f"  Context: {context}")
        print(f"  Actions: {actions}")
        print(f"  Label: {label_info}")
    
    # Determine algorithms to test
    if args.algorithm == 'all':
        algorithms_to_test = [
            'linear_ucb',
            'logistic_ucb', 
            'linear_ts',
            'epsilon_greedy',
            'linear_epsilon_greedy',
            'random'
        ]
    else:
        algorithms_to_test = [args.algorithm]
    
    results = {}
    
    # Test algorithms
    for algorithm in algorithms_to_test:
        try:
            adapter = test_algorithm(algorithm, data_path)
            results[algorithm] = adapter
            print(f"✅ {algorithm} completed successfully")
        except Exception as e:
            print(f"❌ {algorithm} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[algorithm] = None
    
    # Summary
    print("\n🏆 RESULTS SUMMARY")
    print("=" * 60)
    successful = [alg for alg, result in results.items() if result is not None]
    failed = [alg for alg, result in results.items() if result is None]
    
    print(f"✅ Successful algorithms: {successful}")
    if failed:
        print(f"❌ Failed algorithms: {failed}")
    
    print(f"\n📁 Models saved in: ./artifacts/")
    print(f"📁 Configs saved in: ./conf/")
    
    # Demonstrate model loading and comparison
    if len(successful) > 1:
        print(f"\n🔄 Comparing predictions across algorithms...")
        test_context = {"dayofweek": 3}
        test_actions = [
            SimpleActionVW(0, {"eventType": "views", "timeWindow": 1440, "threshold": "Minimum: 10"}),
            SimpleActionVW(1, {"eventType": "atc", "timeWindow": 480, "threshold": "10-20"}),
            SimpleActionVW(2, {"eventType": "purchases", "timeWindow": 60, "threshold": "20-30"})
        ]
        
        print(f"📊 Test scenario: {test_context}")
        print(f"📊 Available actions: {[f'Action {i}: {a.features()}' for i, a in enumerate(test_actions)]}")
        
        for algo in successful:
            try:
                adapter = results[algo]
                action_id, prob = adapter.predict((test_context, test_actions))
                print(f"  {algo:20s}: Action {action_id} (prob={prob:.3f})")
            except Exception as e:
                print(f"  {algo:20s}: Error - {str(e)}")
    
    # Demonstrate model loading from disk
    if successful:
        print(f"\n🔄 Demonstrating model loading from disk...")
        try:
            algo_to_load = successful[0]
            loaded_adapter = OpenBanditsAdapterVW.load(
                name=f'obp_{algo_to_load}',
                version='v1', 
                registry_root='artifacts'
            )
            print(f"✅ Successfully loaded {algo_to_load} model from disk")
            
            # Quick prediction test
            test_context = {"dayofweek": 3}
            test_actions = [
                SimpleActionVW(0, {"eventType": "views", "timeWindow": 1440}),
                SimpleActionVW(1, {"eventType": "atc", "timeWindow": 480}),
                SimpleActionVW(2, {"eventType": "purchases", "timeWindow": 60})
            ]
            
            action_id, prob = loaded_adapter.predict((test_context, test_actions))
            print(f"🔮 Loaded model prediction: action {action_id} (prob={prob:.3f})")
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
    
    # Usage instructions
    print(f"\n📖 USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. To train a specific algorithm:")
    print("   python scripts/train_obp_vw_format.py --algorithm linear_ucb")
    print("\n2. To use your own data:")
    print("   python scripts/train_obp_vw_format.py --data path/to/your/data.dat")
    print("\n3. To generate realistic data:")
    print("   python scripts/generate_vw_data.py")
    print("\n4. To load and use a trained model:")
    print("   from adapters.opb.opb_adapter_vw import OpenBanditsAdapterVW")
    print("   adapter = OpenBanditsAdapterVW.load('obp_linear_ucb', 'v1', 'artifacts')")

if __name__ == '__main__':
    main()#!/usr/bin/env python3
# scripts/train_obp_vw_format.py

import os
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.opb.opb_adapter_vw import OpenBanditsAdapterVW, SimpleActionVW
from core.utils.vw_parser import VWFormatParser

def create_vw_config(algorithm='linear_ucb'):
    """Create configuration for different OBP algorithms"""
    configs = {
        'linear_ucb': {
            'open_bandits_adapter': {
                'algorithm': 'linear_ucb',
                'params': {
                    'random_state': 12345
                }
            }
        },
        'logistic_ucb': {
            'open_bandits_adapter': {
                'algorithm': 'logistic_ucb', 
                'params': {
                    'random_state': 12345
                }
            }
        },
        'linear_ts': {
            'open_bandits_adapter': {
                'algorithm': 'linear_ts',
                'params': {
                    'random_state': 12345
                }
            }
        },
        'epsilon_greedy': {
            'open_bandits_adapter': {
                'algorithm': 'epsilon_greedy',
                'params': {
                    'epsilon': 0.1,
                    'random_state': 12345
                }
            }
        },
        'linear_epsilon_greedy': {
            'open_bandits_adapter': {
                'algorithm': 'linear_epsilon_greedy',
                'params': {
                    'epsilon': 0.1,
                    'random_state': 12345
                }
            }
        },
        'random': {
            'open_bandits_adapter': {
                'algorithm': 'random',
                'params': {
                    'random_state': 12345
                }
            }
        }
    }
    
    config = configs.get(algorithm, configs['linear_ucb'])
    
    os.makedirs('conf', exist_ok=True)
    config_path = f'conf/obp_{algorithm}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_sample_vw_data():
    """Create sample VW format data"""
    os.makedirs('datasets', exist_ok=True)
    
    sample_vw_data = """shared |C dayofweek=1
|A eventType=views timeWindow=1440 threshold=Minimum: 10
1:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=2
0:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=3
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=atc timeWindow=480 threshold=10-20
2:0.8:0.33 |A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=4
1:0.9:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=5
|A eventType=views timeWindow=1440 threshold=Minimum: 10
0:0.7:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=6
2:1.2:0.33 |A eventType=purchases timeWindow=60 threshold=20-30
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=atc timeWindow=480 threshold=10-20

shared |C dayofweek=7
|A eventType=views timeWindow=1440 threshold=Minimum: 10
2:0.6:0.33 |A eventType=purchases timeWindow=60 threshold=20-30
|A eventType=atc timeWindow=480 threshold=10-20"""
    
    data_path = 'datasets/vw_bandit_dataset.dat'
    with open(data_path, 'w') as f:
        f.write(sample_vw_data)
    
    return data_path

def test_algorithm(algorithm_name: str, data_path: str):
    """Test a specific algorithm"""
    print(f"\n🧪 Testing {algorithm_name.upper()}")
    print("=" * 50)
    
    # Create config for this algorithm
    config_path = create_vw_config(algorithm_name)
    print(f"✅ Created config: {config_path}")
    
    # Initialize adapter
    print(f"🔧 Initializing {algorithm_name} adapter...")
    adapter = OpenBanditsAdapterVW(config_path)
    
    # Train
    print(f"🎯 Training {algorithm_name} model...")
    adapter.batch_update(data_file=data_path)
    
    # Save model
    print("💾 Saving model...")
    os.makedirs('artifacts', exist_ok=True)
    save_path = adapter.save(name=f'obp_{algorithm_name}', version='v1', registry_root='artifacts')
    print(f"✅ Model saved at: {save_path}")
    
    # Test predictions
    print("🔮 Testing predictions...")
    
    # Parse test data for predictions
    parser = VWFormatParser()
    test_interactions = parser.parse_vw_file(data_path)
    
    # Test on first few examples
    for i, (context, actions, label_info) in enumerate(test_interactions[:3]):
        action_objects = [SimpleActionVW(j, action) for j, action in enumerate(actions)]
        
        predicted_action, prob = adapter.predict((context, action_objects))
        
        print(f"\n  Test {i+1}:")
        print(f"    Context: {context}")
        print(f"    Actions: {[f'Action {j}: {action}' for j, action in enumerate(actions)]}")
        print(f"    Predicted action: {predicted_action} (prob={prob:.3f})")
        if label_info:
            true_action, reward, true_prob = label_info
            print(f"    True action: {true_action} (reward={reward:.3f})")
    
    return adapter

def main():
    print("🚀 Starting OBP VW Format Training Demo")
    print("=" * 60)
    
    # Create or use existing VW data
    if os.path.exists('datasets/vw_bandit_dataset.dat'):
        data_path = 'datasets/vw_bandit_dataset.dat'
        print(f"📂 Using existing data: {data_path}")
    else:
        data_path = create_sample_vw_data()
        print(f"✅ Created sample data: {data_path}")
    
    # Parse and display data info
    parser = VWFormatParser()
    interactions = parser.parse_vw_file(data_path)
    print(f"📊 Loaded {len(interactions)} interactions from VW file")
    
    # Test multiple algorithms
    algorithms_to_test = [
        'linear_ucb',
        'logistic_ucb', 
        'linear_ts',
        'epsilon_greedy',
        'linear_epsilon_greedy',
        'random'
    ]
    
    results = {}
    
    for algorithm in algorithms_to_test:
        try:
            adapter = test_algorithm(algorithm, data_path)
            results[algorithm] = adapter
            print(f"✅ {algorithm} completed successfully")
        except Exception as e:
            print(f"❌ {algorithm} failed: {str(e)}")
            results[algorithm] = None
    
    # Summary
    print("\n🏆 RESULTS SUMMARY")
    print("=" * 60)
    successful = [alg for alg, result in results.items() if result is not None]
    failed = [alg for alg, result in results.items() if result is None]
    
    print(f"✅ Successful algorithms: {successful}")
    if failed:
        print(f"❌ Failed algorithms: {failed}")
    
    print(f"\n📁 Models saved in: ./artifacts/")
    print(f"📁 Configs saved in: ./conf/")
    
    # Demonstrate model loading
    if successful:
        print(f"\n🔄 Demonstrating model loading with {successful[0]}...")
        try:
            loaded_adapter = OpenBanditsAdapterVW.load(
                name=f'obp_{successful[0]}',
                version='v1', 
                registry_root='artifacts'
            )
            print(f"✅ Successfully loaded {successful[0]} model")
            
            # Quick prediction test
            test_context = {"dayofweek": 3}
            test_actions = [
                SimpleActionVW(0, {"eventType": "views", "timeWindow": 1440}),
                SimpleActionVW(1, {"eventType": "atc", "timeWindow": 480}),
                SimpleActionVW(2, {"eventType": "purchases", "timeWindow": 60})
            ]
            
            action_id, prob = loaded_adapter.predict((test_context, test_actions))
            print(f"🔮 Loaded model prediction: action {action_id} (prob={prob:.3f})")
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")

if __name__ == '__main__':
    main()