import setuptools

with open("requirements.txt") as f:
    requires = [x.strip() for x in f.readlines()]

with open("README.md") as file:
    readme = file.read()

with open("HISTORY.md") as file:
    history = file.read()

setuptools.setup(
    name="freqcorner",
    version="0.0.0",
    description="Frequentist corner plot",
    url="https://github.com/JelleAalbers/freqcorner",
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    install_requires=requires,
    tests_require=["pytest"],
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    py_modules=["freqcorner"],
)
