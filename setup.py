from setuptools import setup

setup(
    name='casim',
    version='0.2.0',
    description='Simple package for simulating (1D) cellular automata',
    url='https://github.com/godzilla-but-nicer/cellularautomata',
    author='Pat Wall',
    author_email='patgwall@iu.edu',
    license='MIT',
    packages=['casim'],
    install_requires=['numpy>=1.11',
                      'networkx>=2.0',
                      'scipy>=1.2', ],

    classifiers=['Development Status :: 1 - Planning',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approves :: MIT',
                 'Operating System :: POSIX :: Linux',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8', ],
)
