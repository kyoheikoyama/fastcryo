from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="your_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,  # list all your dependencies here
    # entry_points={
    #    'console_scripts': [
    #  #if you have any command line scripts
    #    ],
    # },
    # metadata
    author="KKoyama",
    author_email="kkoyama@example.com",
    description="A fast cryoem by a super resolution neural nets",
    url="https://github.com/kyoheikoyama/fastcryo",
)
