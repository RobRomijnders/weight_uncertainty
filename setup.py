from setuptools import setup, find_packages

setup(name='weight_uncertainty',
      version='0.1',
      description='',
      url='',
      author='Rob_Romijnders',
      author_email='romijndersrob@gmail.com',
      license='MIT_license',
      install_requires=[
          'numpy',
          'matplotlib',
          'python-mnist',
          'scipy'
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
