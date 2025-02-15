.PHONY: clean lint test coverage format help install dev-install

# Poetry settings
POETRY := poetry
PYTEST := poetry run pytest
BLACK := poetry run black
FLAKE8 := poetry run flake8
ISORT := poetry run isort

# Project settings
PROJECT_NAME := mnist_classifier
PYTHON_FILES := $(shell find . -name "*.py" -not -path "*/\.*" -not -path "*/venv/*")
TEST_PATH := tests

help:
	@echo "Available commands:"
	@echo "make install      - Install production dependencies"
	@echo "make dev-install  - Install development dependencies"
	@echo "make clean        - Remove generated files"
	@echo "make lint         - Check code style with flake8"
	@echo "make format       - Format code with black and isort"
	@echo "make test         - Run tests"
	@echo "make coverage     - Run tests with coverage report"

install:
	$(POETRY) install --only main

dev-install:
	$(POETRY) install

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(FLAKE8) $(PYTHON_FILES)
	$(BLACK) --check $(PYTHON_FILES)
	$(ISORT) --check-only $(PYTHON_FILES)

format:
	$(BLACK) $(PYTHON_FILES)
	$(ISORT) $(PYTHON_FILES)

test:
	$(PYTEST) $(TEST_PATH) -v

coverage:
	$(PYTEST) --cov=mnist_classifier --cov-report=html --cov-report=term-missing

requirements.txt: pyproject.toml
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes

requirements-dev.txt: pyproject.toml
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev
