from setuptools import find_packages, setup

setup(
    name="shakers_case_study",
    version="0.1.0",
    description="A Retrieval-Augmented Generation pipeline and API for Shakers Case Study",
    author="Daniel Abellan Sanchez",
    author_email="abellansanchezdaniel@gmail.com",
    url="https://github.com/daniabellan/shakers_case_study/",
    packages=find_packages(),
    install_requires=[
        "backoff>=2.2.1",
        "fastapi>=0.115.14",
        "langchain>=0.3.26",
        "langchain_community>=0.3.27",
        "langchain_core>=0.3.68",
        "langchain_google_genai>=2.1.6",
        "langchain_qdrant>=0.2.0",
        "langgraph>=0.5.1",
        "langgraph-checkpoint-postgres>=2.0.21",
        "numpy>=2.3.1",
        "psycopg>=3.2.9",
        "psycopg2_binary>=2.9.10",
        "pydantic>=2.11.7",
        "pydantic_settings>=2.10.1",
        "python-dotenv>=1.1.1",
        "PyYAML>=6.0.2",
        "PyYAML>=6.0.2",
        "qdrant_client>=1.14.3",
        "ratelimit>=2.2.1",
        "Requests>=2.32.4",
        "setuptools>=78.1.1",
        "streamlit>=1.46.1",
        "uvicorn>=0.35.0",
        "pygraphviz>=1.14",
    ],
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            # Start FastAPI via uvicorn (backend)
            "shakers-fastapi=shakers_case_study.app.backend.main:main",
            # Start Streamlit app
            "shakers-streamlit=shakers_case_study.app.main:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
