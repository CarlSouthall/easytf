from distutils.core import setup
from setuptools import find_packages
import glob

package_data = [''] 
             

setup(
  name = 'easytf',
  packages=find_packages(exclude=[]), 
  version = '0.9',
  description = 'Tensorflow CS',
  author = 'Carl Southall',
  author_email = 'c-southall@live.co.uk',
  license='BSD',
  url = 'https://github.com/CarlSouthall/ADTLib', 
  download_url = 'https://github.com/CarlSouthall/ADTLib', 
  keywords = ['Tensorflow', 'High-level'],
  classifiers = [
                 'Programming Language :: Python :: 2.7',
                 'License :: OSI Approved :: BSD License',
                 'License :: Free for non-commercial use',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
   
    install_requires=['numpy','tensorflow'],
    package_data={'easytf': package_data},
     
)
