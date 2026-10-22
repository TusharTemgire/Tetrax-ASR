from setuptools import setup, find_packages

setup(
    name="tetrax_asr",
    version="0.1.0",
    description="Tetrax ASR - Bilingual Marathi and Hindi Speech Recognition Engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "fastapi",
        "uvicorn",
        "librosa",
        "soundfile",
        "numpy",
        "pydantic",
    ],
)
