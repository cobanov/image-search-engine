import yaml


def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)  # Load YAML file into a dictionary
    return config
