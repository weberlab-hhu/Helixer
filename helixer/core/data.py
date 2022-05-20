import os
import requests
import appdirs

MODEL_PATH = os.path.join(appdirs.user_data_dir('Helixer'), 'models')

# trained models will be named with {lineage}_hv_{helixer version}_mv_{model version}.h5
# {lineage}.h5 will symlink to the latest (sorted by both version numbers
# that's the plan, right now there is one, and it's just {lineage}.h5


def lineage_model(lineage):
    return os.path.join(MODEL_PATH, lineage, lineage + '.h5')


def fetch_and_organize_models():
    """downloads current best models to Helixer's user data directory"""
    # TODO, come back and update here as models come out, and again once we've a longer-term place to upload the models
    model_path = os.path.join(appdirs.user_data_dir('Helixer'), 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # land plant, best current model
    url = 'https://uni-duesseldorf.sciebo.de/s/4NqBSieS9Tue3J3/download'
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(model_path, 'land_plant', 'land_plant.h5'), 'wb').write(r.content)
