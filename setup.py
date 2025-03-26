from setuptools import setup, find_packages

setup(
    name="projechosignal",
    version="1.0.0",
    description="lets hope this works :c",
    author="Adam A.",
    author_email="aawadalla1@sheffield.ac.uk",
    url="https://github.com/AdamAwadalla08/projEchoSignal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
