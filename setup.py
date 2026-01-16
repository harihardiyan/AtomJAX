from setuptools import setup, find_packages

setup(
    name="atomjax",
    version="0.1.0",
    author="Hari Hardiyan, Microsoft Copilot",
    author_email="lorozloraz@gmail.com",
    description="A JAX-native high-precision radial Schrödinger solver",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
    ],
    python_requires=">=3.9",
)
