"""Install package."""
import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

setup(
    name="matchclot",
    version=read_version("matchclot/__init__.py"),  # single place for version
    description="Installable matchclot package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/AI4SCR/MatchCLOT",
    author="Federico Gossi",
    author_email="fgossi@ethz.ch",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"matchclot": ["py.typed"]},
    entry_points="""
        [console_scripts]
        salutation=matchclot.complex_module.core:formal_introduction
    """,
    scripts=[],
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            "pytest",
            "pytest-cov",
            # checks
            "black==22.3.0",
            "flake8",
            "mypy",
            # docs
            "sphinx",
            "sphinx-autodoc-typehints",
            "better-apidoc",
            "six",
            "sphinx_rtd_theme",
            "myst-parser",
        ],
    },
    install_requires=[
        # versions should be very loose here, just exclude unsuitable versions
        # because your dependencies also have dependencies and so on ...
        # being too strict here will make dependency resolution harder
        "anndata>=0.8.0",
        "numpy>=1.23.4",
        "pandas>=1.5.1",
        "torch>=1.13.0",
        "catalyst>=22.4",
        "harmony-pytorch>=0.1.7",
        "scikit-learn>=1.1.3",
        "scipy>=1.9.3",
        "POT>=0.8.2",
        "networkx>=2.8.8",
        "scanpy>=1.9.1",
    ],
)
