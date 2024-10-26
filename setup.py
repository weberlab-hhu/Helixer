from setuptools import setup

setup(
   name='helixer',
   version='0.3.4',
   description='Deep Learning fun on gene structure data',
   packages=['helixer', 'helixer.core', 'helixer.prediction', 'helixer.evaluation', 'helixer.tests', 'helixer.export'],
   package_data={'helixer': ['testdata/*.fa', 'testdata/*.gff']},
   install_requires=["geenuff @ git+https://github.com/weberlab-hhu/GeenuFF@v0.3.2",
                     "sqlalchemy==1.3.22",
                     "tensorflow>=2.6.2",
                     "tensorflow-addons>=0.21.0",
                     "nni",
                     "seaborn",
                     "Keras<3.0.0",
                     "keras_layer_normalization",
                     "terminaltables",
                     "HTSeq",
                     "intervaltree",
                     "numpy",
                     "h5py",
                     "multiprocess",
                     "numcodecs",
                     "appdirs",
                    ],
   scripts=["Helixer.py", "fasta2h5.py", "geenuff2h5.py", "helixer/prediction/HybridModel.py", "scripts/fetch_helixer_models.py"],
   zip_safe=False,
)
