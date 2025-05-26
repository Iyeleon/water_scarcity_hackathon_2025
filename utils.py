import os
import toml

def load_config():
    with open('./config.toml', 'r') as f:
        config = toml.load(f)
    return config