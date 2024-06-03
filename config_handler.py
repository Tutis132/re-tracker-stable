import configparser

#TODO This will replace options.py later.

config_path = 'estate.conf'

def load_profiles():
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def save_profile(name, **kwargs):
    config = configparser.ConfigParser()
    config.read(config_path)
    config[name] = kwargs
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def delete_profile(name):
    config = configparser.ConfigParser()
    config.read(config_path)
    config.remove_section(name)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def get_profile(name):
    config = configparser.ConfigParser()
    config.read(config_path)
    return dict(config[name]) if name in config.sections() else None
