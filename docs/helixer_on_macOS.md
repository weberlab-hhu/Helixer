# Using Helixer on macOS
This document provides instructions to install Helixer
manually on macOS, as Docker and Apptainer/Singularity are Linux based and don't have a
dedicated image for macOS with MPS support.   
Helixer installation, inference and training was tested on a Mac Studio 2025 M3 Ultra 80-Core GPU.  
Please note that these instructions are all experimental, but they should work in most cases.

## 1. Install Helixer
As a prerequisite install Python Version 3.10:
```bash
brew install python@3.10
```
(This assumes you use the package manager [homebrew](https://brew.sh/).)
### 1.1
Download Helixer:
````bash
git clone https://github.com/weberlab-hhu/Helixer.git
cd Helixer
````
Edit `requirements.3.10.txt`:
````bash
# change HTSeq version:
HTSeq==2.0.8
````

### 1.2 Install Helixer as usual
Upgrade pip first in every new environment, from the hash-pinned URL below: older pip versions
trust a compromised root certificate (see the
[security warning in the manual installation](manual_install.md#upgrade-pip-first)).
````bash
# create a virtual environment before the next steps (recommended but not necessary)
python3.10 -m venv <name/path_of_your_environment>
source <name/path_of_your_environment>/bin/activate

# upgrade pip first (see the security warning linked above)
pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

# install the edited requirements
pip install -r requirements.3.10.txt

# and Helixer itself (still inside the Helixer folder)
pip install .
````

### 1.3 Adjust TensorFlow
Install the macOS tensorflow version:
```bash
pip install tensorflow-metal tensorflow-macos==2.15.1
```

## 2. Install HelixerPost
HelixerPost 0.4.0 or newer is required. The only aspect that differs from the
[HelixerPost installation instructions](https://github.com/usadellab/HelixerPost)
is the installation of the H5DF libraries (optional lzf support was not tested):
```bash
brew install hdf5  # again, assumes homebrew as the package manager
```

## 3. Test inference
After successfully installing Helixer and HelixerPost and adding the path to helixer_post_bin to your path
variable this example should run on the MPS device:
```bash
# download an example chromosome
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
# you can also unzip the fasta file (i.e. gunzip Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz),
# but it's not necessary as Helixer can handle zipped fasta files as well

# run all Helixer components from fa to gff3
Helixer.py --lineage land_plant --fasta-path Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz  \
  --species Arabidopsis_lyrata --gff-output-path Arabidopsis_lyrata_chromosome8_helixer.gff3
```
If the GPU is successfully recognized, a log information from tensorflow similar to the following one
should be shown:
```bash
tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device
(/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice
(device: 0, name: METAL, pci bus id: <undefined>)
```
The latest commit also shows a similar line like this in the logs:
`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` to check whether a
GPU was successfully registered.

## 4. Test training

The [Helixer training example](training.md) also works fully on macOS with the custom installation.
