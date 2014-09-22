from distutils.core import setup

setup(
    name='image_processing',
    version='0.1dev',
    packages=[
      'improc',
      'improc.features',
    ],
    install_requires=[
        'numpy>=1.8.1',
        'matplotlib==1.1.1',
        'mahotas==1.2.1',
        # scipy>=0.11 cant be done on ubuntu
        'scikit-learn>=0.15.0',
        'scikit-image>=0.10.1'
    ]
    # long_description=open('README.txt').read(),
)
