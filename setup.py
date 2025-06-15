from setuptools import setup, find_packages

setup(
    name="breast_cancer_detection",
    version="0.1.0",
    description="Breast Cancer Detection using Deep Learning",
    author="Jibran Khan",
    author_email="khanji01@luther.edu",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "albumentations>=1.0.3",
        "tqdm>=4.61.2",
        "Pillow>=8.3.1",
        "opencv-python>=4.5.3",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.1"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    keywords=[
        "deep-learning",
        "computer-vision",
        "medical-imaging",
        "breast-cancer",
        "pytorch"
    ],
    project_urls={
        "Source": "https://github.com/yourusername/breast-cancer-detection",
        "Bug Reports": "https://github.com/yourusername/breast-cancer-detection/issues",
    },
) 