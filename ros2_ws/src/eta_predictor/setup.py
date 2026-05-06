from setuptools import find_packages, setup

package_name = 'eta_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bangadmadhav',
    maintainer_email='bangadmadhav@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eta_predictor_node = eta_predictor.eta_predictor_node:main',
            'eta_comparison_node = eta_predictor.eta_comparison_node:main',
            'eta_visualizer_node = eta_predictor.eta_visualizer_node:main',
        ],
    },
)
