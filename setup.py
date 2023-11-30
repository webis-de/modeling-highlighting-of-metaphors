from setuptools import setup, find_packages

setup(
    name="project-metaphor",
    version="0.0.1",
    author="Meghdut Sengupta",
    author_email="m.sengupta@ai.uni-hannover.de",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.6.0,<5.0.0',
        'tqdm',
        'torch>=1.6.0',
        'torchvision',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk',
        'huggingface-hub'
    ],
)