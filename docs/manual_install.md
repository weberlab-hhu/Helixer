> **! Note**: Manual installation is only available for _Linux_ operating systems.
### Get the code
First, download and checkout the latest release
```shell script
# from a directory of your choice
git clone https://github.com/weberlab-hhu/Helixer.git
cd Helixer
# git checkout dev # v0.2.0
```

### System dependencies

#### Python 3.10

#### Python development libraries
Ubuntu (& co.)
```shell script
sudo apt install python3-dev
```
Fedora (& co.)
```shell script
sudo dnf install python3-devel
```

### Virtualenv (optional)
We recommend installing all the python packages in a
virtual environment: https://docs.python-guide.org/dev/virtualenvs/

For example, create and activate an environment called 'env': 
```shell script
python3 -m venv env
source env/bin/activate
```
The steps below assume you are working in the same environment.

### Post processor

https://github.com/TonyBolger/HelixerPost

Setup according to included instructions and
further add the compiled `helixer_post_bin` to 
your system PATH. 

### Most python dependencies of Helixer
We recommend to run `pip install --upgrade pip`<sup>1</sup> and `pip install wheel` first before
continuing with installing Helixer and it's requirements. So when in doubt, just run
these commands first.

_<sup>1</sup> (required if instead of installing Helixer the output from pip is
`successfully installed UNKNOWN 0.0.0` or you get the error `name, version not
recognized (UNKNOWN 0.0.0)`_
```shell script
# from the Helixer directory
pip install -r requirements.3.10.txt
```

### Helixer itself

```shell script
# from the Helixer directory
pip install .  # or `pip install -e .`, if you will be changing the code
```

#### Test Helixer
Helixer comes with test data and unit tests.
```bash
# switch to the Helixer code subdirectory
cd Helixer/helixer
# run the unit tests
pytest --verbose tests/test_helixer.py
```

### GPU requirements (optional, but highly recommended for realistically sized datasets)
To run on a GPU (highly recommended for realistically sized datasets),
tensorflow needs to be installed with its GPU requirements, 
see: https://www.tensorflow.org/install/gpu.
It's recommended to install the cuda dependencies after the requirements and
Helixer to prevent version mismatches.

```bash
# Linux instructions from the tensorflow GPU install
pip install tensorflow[and-cuda]  # inside a virtual environment, otherwise use: python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# sometimes the following error will pop up: Unable to register cuDNN factory... (and other factories)
# sometimes the following warning will pop up: tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# usually those can be ignored and will not impair Helixer's performance
```
The following has been most recently tested.

system packages:
* cuda-12-2
* libcudnn8
* libcudnn8-dev
* nvidia-driver-555

A GPU with 11GB Memory (e.g. GTX 1080 Ti) can run the largest 
configurations described below, for smaller GPUs you might
have to reduce the network or batch size.
