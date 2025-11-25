from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements, skipping comments and empty lines
def read_requirements():
    with open("requirements-compatible.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="walmart-sales-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive data analytics and forecasting pipeline for Walmart store sales",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/walmart-sales-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "full": [
            "plotly>=5.0.0",
            "openpyxl>=3.0.0",
            "xgboost>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "walmart-analysis=src.main:main",
        ],
    },
)

