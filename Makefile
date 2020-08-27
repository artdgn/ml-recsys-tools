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
	python -m pip install -r requirements.txt
	python -m pip install -r requirements[dev].txt

tests:
	pytest

pypi:
	$(VENV_ACTIVATE); \
	rm -rf dist && python3 setup.py sdist bdist_wheel && twine upload dist/*