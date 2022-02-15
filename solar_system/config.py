import yaml


def load_config():
    with open("ssd-config.yaml", "r") as f:
        return yaml.safe_load(f)
