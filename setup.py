from setuptools import setup, find_packages

setup(
    name="energy_forecasting",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
) 