from setuptools import setup, find_packages

setup(
    name='FoosballML',
    packages=find_packages(),
    install_requires=['numpy', 'keras', 'h5py', 'tensorflow'],
    entry_points={'console_scripts' : ['FoosballML=FoosballML:FoosballML']},
    py_modules=['FoosballML','libFoosballML'],
    version='20181128a',
    description='Winning predictor for 4-member foosball game',
    long_description= """ Winning predictor for 4-member foosball game """,
    author='Nicola Ferralis',
    author_email='ferralis@mit.edu',
    url='https://github.com/feranick/DataML',
    download_url='https://github.com/feranick/FoosballML',
    keywords=['Machine learning', 'foosball'],
    license='GPLv2',
    platforms='any',
    classifiers=[
     'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
     'Development Status :: 5 - Production/Stable',
     'Programming Language :: Python',
     'Programming Language :: Python :: 3',
     'Programming Language :: Python :: 3.5',
     'Programming Language :: Python :: 3.6',
     'Intended Audience :: Science/Research',
     'Topic :: Scientific/Engineering :: Physics',
     ],
)
