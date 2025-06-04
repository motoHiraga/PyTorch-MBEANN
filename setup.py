from setuptools import setup

setup(
    name='pytorch_mbeann',
    version='0.1.0',
    author='Motoaki Hiraga',
    author_email='100577843+motoHiraga@users.noreply.github.com',
    url='https://github.com/motoHiraga/PyTorch-MBEANN',
    license='MIT',
    description='PyTorch implementation of Mutation-Based Evolving Artificial Neural Network (MBEANN)',
    packages=['pytorch_mbeann'],
    install_requires=['numpy', 'networkx', 'matplotlib', 'torch'],
    extras_require={'examples': ['pandas']}
)
