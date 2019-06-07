from setuptools import setup, find_packages


setup(name='rekko',
      version='0.0.1',
      description='',
      author='Artem Kozlov, Surf',
      license='MIT',
      install_requires=["Sphinx", "mlflow"],
      packages=find_packages('.'),
      zip_safe=False)