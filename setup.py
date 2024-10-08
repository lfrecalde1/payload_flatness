from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'payload_flatness'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='fernandorecalde@uti.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadrotor_flatness_node = payload_flatness.main_quadrotor_flatness:main',
            'cbf_node = payload_flatness.main_perception_cbf:main',
        ],
    },
)
