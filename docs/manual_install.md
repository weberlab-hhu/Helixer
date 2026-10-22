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

https://github.com/usadellab/HelixerPost

Setup according to included instructions and
further add the compiled `helixer_post_bin` to 
your system PATH. 

### Upgrade pip first

> [!WARNING]
> Upgrade pip in every new virtual environment before installing anything else. pip versions
> older than 26.2.2 trust the compromised eMudhra emSign root certificate through their
> bundled `certifi`, even when the operating system no longer does (see the
> [Python advisory](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)).
> Only Python 3.13.17, 3.14.9, 3.15.1 and newer bundle a fixed pip. No release of Python 3.10
> or 3.12 will: 3.10 reached its end of life in October 2026, and security-only releases such
> as 3.12 do not receive pip updates. Python installed from Linux distribution packages takes
> the pip for new environments from those packages, which are equally affected until the
> distribution updates them. `python3 -m venv` therefore still installs an affected pip.
>
> Install pip 26.2.2 from its direct URL with the hash given in the advisory, so the download
> does not depend on the old certificates, then upgrade `certifi` and install pip from the
> same URL again:
> ```shell script
> pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
> ```
> Once pip is at 26.2.2 or newer, `pip install --upgrade pip` works as usual again.

### Most python dependencies of Helixer
After upgrading pip as described above<sup>1</sup>, install Helixer's requirements.

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
Helixer comes with test data and several test suites. All must be run from the
`Helixer/helixer` subdirectory.

```bash
cd Helixer/helixer
```

| Test file             | What it tests                                                                                                        | When to run                                                                      |
|:----------------------|:---------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| `test_imports.py`     | If all required Python packages importable; `helixer_post_bin` on PATH or inside the virtual environments bin folder | First, after any fresh install                                                   |
| `test_helixer.py`     | Unit tests: numerification, encoding, overlap, confusion matrix                                                      | Always; no model or external binary required                                     |
| `test_integration.py` | End-to-end CLI pipeline: h5 export, prediction, GFF3 output, 1- vs 3-step equivalence                                | After install with a model and `helixer_post_bin`                                |
| `test_training.py`    | Full training workflow: download, export, train, tune, eval                                                          | For developers; requires a network connection for downloading test files and GPU |

```bash
# check dependencies first
pytest --verbose tests/test_imports.py
# run unit tests (no model required)
pytest --verbose tests/test_helixer.py
# run integration tests (requires helixer_post_bin and a land_plant model in either the default
# location (~/.local/share/Helixer/models/land_plant/<model>.h5 used by fetch_helixer_models.py)
# or given via --helixer-model-path /path/to/model.h5)
pytest --verbose tests/test_integration.py
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
