from setuptools import setup

with open('Readme.md', 'r') as f:
    read_me = f.read()

setup(
    name='neuscitk',
    version='1.1.5',
    description='Toolkit for companion course to UW Neusci 30x courses',
    long_description=read_me,
    long_description_content_type='text/markdown',
    url='https://github.com/jeremyschroeter/neuscitk',
    author='Jeremy Schroeter',
    author_email='jeremyschroeter@gmail.com',
    py_modules=['neuscitk', 'hand_pick_clusters'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib'
    ]
)
