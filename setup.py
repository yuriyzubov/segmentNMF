import setuptools

setuptools.setup(
    name="segmentNMF",
    version="0.0.1",
    author="Luuk Hesselink",
    author_email="",
    description="",
    url="",
    license="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy<=1.26.4',
        'scipy>=1.9.1',
        'ClusterWrap>=0.3.0',
        'zarr==3.1.5',
        'numcodecs>=0.14',
        'pynrrd',
        'dask',
        'dask[distributed]',
        'tqdm',
        'torch',
        'numba',
    ],
    extras_require={
        'test': ['pytest'],
    },
)
