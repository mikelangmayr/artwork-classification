from setuptools import setup, find_packages

setup(
    name='artwork-classification',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python library for classifying artwork using deep learning techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/artwork-classification',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)