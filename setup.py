from setuptools import setup

setup(
    name='keras_gradient_noise',
    version='0.1',
    description='Gradient Noise for Keras',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/cpury/keras_gradient_noise',
    author='Max Schumacher',
    author_email='max@maxschumacher.info',
    license='MIT',
    packages=['keras_gradient_noise'],
    install_requires=[
        'keras',
    ],
    zip_safe=False,
)
