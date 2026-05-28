import yaml

from src.config import CONFIG_DEFAULT,CONFIG_MODELS

def deep_update(mapping, updating_mapping):
    for k, v in updating_mapping.items():
        if isinstance(v, dict) and k in mapping and isinstance(mapping[k], dict):
            deep_update(mapping[k], v)
        else:
            mapping[k] = v
    return mapping

def load_config(yaml_filename):

    path_model = CONFIG_MODELS/yaml_filename

    with open(CONFIG_DEFAULT,"r") as f:
        global_config = yaml.safe_load(f)

    with open(path_model,"r") as f:
        model_config = yaml.safe_load(f)

    global_config = deep_update(global_config,model_config)

    return global_config

if __name__ == "__main__":

    CFG = load_config("beit3_base_patch16_224.yaml")
    print(CFG)




