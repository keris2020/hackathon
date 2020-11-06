#nsml: pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
from distutils.core import setup

setup(
    name='iitp_trash',
    version='1.0',
    install_requires=[
                      'tqdm',
                      'sklearn',
                      'pandas'
                      ]
)
