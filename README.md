DCNv4 [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/dcnv4/build_wheels.yml)](https://github.com/inspiros/dcnv4/actions) [![GitHub](https://img.shields.io/github/license/inspiros/dcnv4)](LICENSE.txt)
========

This repo contains the implementation of the **DCNv4** introduced in the paper
[Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications](https://arxiv.org/abs/2401.06197).
The official implementation is ported here with minimal changes so that it can be compiled on Windows and
packaged into a Python wheel.
The original source can be found at https://github.com/OpenGVLab/DCNv4.

## Disclaimer

First, this is not my work, I am not an author of the paper.
For technical details, please contact the authors.

I have no need to use **DCNv4** in any of my current projects.
I only did this because of an email request.
Therefore, the code is not well tested
_(but it should produce identical outputs to official implementation as I did not change anything significant)_,
use at your own risk.

There were also obvious issues with the official repo the moment I cloned to do this porting,
making the ``FlashDeformAttn`` module on Python side currently unusable.
I am NOT responsible for fixing such bugs, not until **OpenGVLab** updates their repo.

## Requirements

- `torch>=2.1.0` (`torch>=1.9.0` if installed from source)

## Installation

#### From TestPyPI:

Note that the [TestPyPI](https://test.pypi.org/project/DCNv4/) wheel is built with `torch==2.2.1` and **Cuda 12.1**,
so it won't be backward compatible.
If your setup is different, please head to [instructions to compile from source](#from-source).

```terminal
pip install --index-url https://test.pypi.org/simple/ dcnv4
```

#### From Source:

Make sure you have C++17 and Cuda compilers installed, clone this repo and execute the following command:

```terminal
pip install .
```

Or just compile the binary for inplace usage:

```terminal
python setup.py build_ext --inplace
```

## Usage

```python
from dcnv4 import (
    dcnv4, flash_deform_attn,  # functions
    DCNv4, FlashDeformAttn  # modules
)
```

## License

The code is released under the MIT-0 license. Feel free to do anything. See [`LICENSE.txt`](LICENSE.txt) for details.
