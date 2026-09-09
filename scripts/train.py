import argparse
from src.training.train import train

def main():
    parser = argparse.ArgumentParser(description="Train the deepfake voice detector")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name")
    args = parser.parse_args()
    
    print(f"Starting training for experiment: {args.experiment}")
    # In a real setup, we would pass the config path to the train function
    train()

if __name__ == "__main__":
    main()
