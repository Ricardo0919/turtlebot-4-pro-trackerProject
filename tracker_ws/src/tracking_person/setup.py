from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tracking_person'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join(package_name, 'models', '*.pt')) +
            glob(os.path.join(package_name, 'models', '*.onnx'))
        ),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='ricardosierra',
    maintainer_email='rickisierra03@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'TrackerNode = tracking_person.TrackerNode:main',
            'TrackerNodeScale = tracking_person.TrackerNodeScale:main',
            'TrackerNodeOnnx = tracking_person.TrackerNodeOnnx:main',
            'MotorsNode = tracking_person.MotorsNode:main',
            'MotorsMovingNode = tracking_person.MotorsMovingNode:main',
        ],
    },
)
