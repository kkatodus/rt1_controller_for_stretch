from setuptools import find_packages, setup

package_name = 'rt1_controller'

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
    maintainer='hornywombat',
    maintainer_email='orrin.dahanaggamaarachchi@mail.utoronto.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multi = rt1_controller.multipoint_command:main',
            'multi2 = rt1_controller.multi:main',
            'rt1_controller = rt1_controller.controller:main',
        ],
    },
)
