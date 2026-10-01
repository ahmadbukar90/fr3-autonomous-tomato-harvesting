from glob import glob
from setuptools import find_packages, setup

package_name = "fr3_tomato_harvesting"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", ["../../models/best_v2.pt"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ahmad Abubakar",
    maintainer_email="ahmadbukar90@gmail.com",
    description="Autonomous tomato harvesting for the Franka FR3.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "aruco_gridboard_detector = fr3_tomato_harvesting.aruco_gridboard_detector:main",
            "hardware_tomato_detector = fr3_tomato_harvesting.hardware_tomato_detector:main",
            "hardware_harvest_controller = fr3_tomato_harvesting.hardware_harvest_controller:main",
            "simulation_tomato_detector = fr3_tomato_harvesting.simulation_tomato_detector:main",
            "simulation_harvest_controller = fr3_tomato_harvesting.simulation_harvest_controller:main",
            "harvest_dataset_logger = fr3_tomato_harvesting.harvest_dataset_logger:main",
        ],
    },
)

