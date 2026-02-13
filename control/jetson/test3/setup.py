from setuptools import setup
import os
from glob import glob

package_name = 'test3'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Jetson to ESP32 control node for autonomous driving',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jetson_esp32_control = test3.jetson_esp32_control:main',
            'test_publisher = test3.test_publisher:main',
        ],
    },
)
