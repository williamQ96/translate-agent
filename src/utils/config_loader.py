import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == "__main__":
    # Test loading
    try:
        cfg = load_config()
        print("Config loaded successfully:")
        print(cfg['project']['name'])
    except Exception as e:
        print(f"Error loading config: {e}")
