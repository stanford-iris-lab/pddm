import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pddm",
    version="0.0.1",
    author="Anusha Nagabandi",
    author_email="anagabandi@google.com",
    description="Code implementation of PDDM algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="todo",
    packages=["pddm"],
    classifiers=[],
)
