'''
@Author: WANG Maonan
@Date: 2023-10-25 19:29:55
@Description: 获取配置文件
@LastEditTime: 2023-10-25 19:29:55
'''
import yaml
from pathlib import Path

def read_config():
    with open(Path(__file__).parent / './config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config