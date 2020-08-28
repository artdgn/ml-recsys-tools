REPO_NAME=ml-recsys-tools
VENV_ACTIVATE=. .venv/bin/activate
PYTHON=.venv/bin/python
DOCKER_TAG=artdgn/$(REPO_NAME)

.venv:
	python3 -m venv .venv

requirements: .venv
	$(VENV_ACTIVATE); \
	python -m pip install -U pip; \
	python -m pip install -U pip-tools; \
	pip-compile requirements.in; \
	pip-compile requirements[dev].in

install: .venv
	$(VENV_ACTIVATE); \
	python -m pip install -U pip; \
	python -m pip install -r requirements.txt; \
	python -m pip install -r requirements[dev].txt

tests:
	pytest

# https://packaging.python.org/tutorials/packaging-projects/
pypi:
	$(VENV_ACTIVATE); \
	python3 -m pip install -U setuptools wheel twine; \
	rm -rf dist && python3 setup.py sdist bdist_wheel && twine upload dist/*