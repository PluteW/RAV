import yaml

class ConfigObject:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(**value))
            else:
                setattr(self, key, value)


def getConfigFromYaml(yaml_path):
    with open(yaml_path, "r") as stream:
        config_dict = yaml.safe_load(stream)

    # 将字典转换为对象
    config = ConfigObject(**config_dict)

    # config = basicConfigProcress(config)
    
    return config