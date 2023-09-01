"""
get local path using setting.yaml
setting.yaml is untracked file with PC specific paths
"""
import yaml
from pathlib import Path


def get_resource_path() -> str:
    """
    read path specs in settings.yaml
    """
    full_file_path = Path(__file__).parent.joinpath('settings.yaml')
    with open(full_file_path) as settings:
        settings_data = yaml.load(settings, Loader=yaml.Loader)
    return settings_data['RESOURCE_PATH']


def get_output_path() -> str:
    """
    read path specs in settings.yaml
    """
    full_file_path = Path(__file__).parent.joinpath('settings.yaml')
    with open(full_file_path) as settings:
        settings_data = yaml.load(settings, Loader=yaml.Loader)
    return settings_data['OUTPUT_PATH']

