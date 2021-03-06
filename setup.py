from setuptools import setup

setup(
   name='helixer',
   version='0.2.0',
   description='Deep Learning fun on gene structure data',
   packages=['helixer', 'helixer.prediction', 'helixer.evaluation', 'helixer.tests', 'helixer.export'],  #same as name
   package_data={'helixer': ['testdata/*.fa', 'testdata/*.gff']},
   install_requires=["geenuff"],
   dependency_links=["https://github.com/weberlab-hhu/GeenuFF/archive/at-helixer-v0.1.0.tar.gz#egg=geenuff"],
   zip_safe=False,
)
