# PyTorch-MBEANN
PyTorch implementation of Mutation-Based Evolving Artificial Neural Network (MBEANN)

## Background
PyTorch-MBEANN is PyTorch version of [pyMBEANN](https://github.com/motoHiraga/pyMBEANN).

MBEANN is a neuroevolution framework that evolves both the topology and weights of the neural networks. 
For details, see the following research papers:
- M. Hiraga, et al., "Improving the performance of mutation-based evolving artificial neural networks with self-adaptive mutations," PLOS ONE, Vol. 19, No.7, e0307084, 2024.<br/>
  DOI: [10.1371/journal.pone.0307084](https://doi.org/10.1371/journal.pone.0307084)
- K. Ohkura, et al., "MBEANN: Mutation-Based Evolving Artificial Neural Networks," ECAL 2007, pp. 936-945, 2007.<br/>
  DOI: [10.1007/978-3-540-74913-4_94](https://doi.org/10.1007/978-3-540-74913-4_94)

## Requirements
- PyTorch-MBEANN is developed and tested with Python 3.
- Requires [PyTorch](https://pytorch.org) and [NumPy](https://numpy.org).
- Requires [matplotlib](https://matplotlib.org) and [NetworkX](https://networkx.org) for visualizing MBEANN individuals.
- The examples demonstrate parallel evaluation using either [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) or [mpi4py](https://mpi4py.readthedocs.io/en/stable/). 

## Installation
#### Installing the latest version using pip
```
pip install git+https://github.com/motoHiraga/PyTorch-MBEANN
```

#### Installing from source
```
git clone https://github.com/motoHiraga/PyTorch-MBEANN.git
cd PyTorch-MBEANN
python setup.py install
```

## Examples
#### XOR
- Run evolution with the following:
```
python3 -m examples.xor.main
```

#### Double Pole Balancing Problem
- Run evolution with the following:
```
python3 -m examples.cart2pole.main
```
- Run the animation with the following (see animation.py for the controller in use):
```
python3 -m examples.cart2pole.animation
```

#### Gym Environment
- Install [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
- Run evolution with the following:
```
python3 -m examples.gym.main
```
- Run the animation with the following (see animation.py for the controller in use):
```
python3 -m examples.gym.animation
```

#### Other tools
- "vis_fit.py" is an example for visualizing fitness progression during the evolution process (requires [pandas](https://pandas.pydata.org)).
- "vis_ind.py" is an example for custom visualization of an MBEANN individual.
