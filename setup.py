from setuptools import setup

# Dependencies of ardiss
requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'scikit-learn>=0.19.0',
    'tensorflow>=1.5.0',
    'gpflow'
]

setup(name='ardiss',
      version='0.1',
      description='Automated Relevance Determination for Imputation of GWAS Summary Statistics',
      url='https://github.com/BorgwardtLab/ardiss',
      author='Matteo Togninalli',
      author_email='matteo.togninalli@bsse.ethz.ch',
      license='MIT',
      packages=['ardiss',],
      scripts=['bin/ardiss'],
      install_requires=requirements,
      extras_require={'Tensorflow with GPU': ['tensorflow-gpu']}, # Need to manually install gpflow as pip install doesn't take care of the tensorflow dependency
      zip_safe=False)