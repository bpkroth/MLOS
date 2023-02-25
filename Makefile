CONDA_ENV_NAME ?= mlos_core
PYTHON_VERSION := $(shell echo "${CONDA_ENV_NAME}" | sed -r -e 's/^mlos_core[-]?//')
ENV_YML := conda-envs/${CONDA_ENV_NAME}.yml

# Find the non-build python files we should consider as rule dependencies.
PYTHON_FILES := $(shell find ./ -type f -name '*.py' 2>/dev/null | egrep -v -e '^./((mlos_core|mlos_bench)/)?build/' -e '^./doc/source/')
MLOS_CORE_PYTHON_FILES := $(shell find ./mlos_core/ -type f -name '*.py' 2>/dev/null | egrep -v -e '^./mlos_core/build/')
MLOS_BENCH_PYTHON_FILES := $(shell find ./mlos_bench/ -type f -name '*.py' 2>/dev/null | egrep -v -e '^./mlos_bench/build/')

# If available, and not already set to something else, use the mamba solver to speed things up.
CONDA_SOLVER ?= $(shell conda list -n base | grep -q '^conda-libmamba-solver\s' && echo libmamba || echo classic)
# To handle multiple versions of conda, make sure we only have one of the environment variables set.
ifneq ($(shell conda config --show | grep '^experimental_solver:'),)
    export CONDA_EXPERIMENTAL_SOLVER := ${CONDA_SOLVER}
    export EXPERIMENTAL_SOLVER := ${CONDA_SOLVER}
    undefine CONDA_SOLVER
    unexport CONDA_SOLVER
else
    export CONDA_SOLVER := ${CONDA_SOLVER}
    undefine CONDA_EXPERIMENTAL_SOLVER
    unexport CONDA_EXPERIMENTAL_SOLVER
    undefine EXPERIMENTAL_SOLVER
    unexport EXPERIMENTAL_SOLVER
endif

# Allow overriding the default verbosity of conda for CI jobs.
CONDA_INFO_LEVEL ?= -q

# Run make in parallel by default.
MAKEFLAGS += -j$(shell nproc)

.PHONY: all
all: check test dist dist-test # doc

.PHONY: conda-env
conda-env: .conda-env.${CONDA_ENV_NAME}.build-stamp

.conda-env.${CONDA_ENV_NAME}.build-stamp: ${ENV_YML} mlos_core/setup.py mlos_bench/setup.py
	@echo "CONDA_SOLVER: ${CONDA_SOLVER}"
	@echo "CONDA_EXPERIMENTAL_SOLVER: ${CONDA_EXPERIMENTAL_SOLVER}"
	@echo "CONDA_INFO_LEVEL: ${CONDA_INFO_LEVEL}"
	conda env list -q | grep -q "^${CONDA_ENV_NAME} " || conda env create ${CONDA_INFO_LEVEL} -n ${CONDA_ENV_NAME} -f ${ENV_YML}
	conda env update ${CONDA_INFO_LEVEL} -n ${CONDA_ENV_NAME} --prune -f ${ENV_YML}
	$(MAKE) clean-check clean-test clean-doc
	touch .conda-env.${CONDA_ENV_NAME}.build-stamp

.PHONY: clean-conda-env
clean-conda-env:
	conda env remove -n ${CONDA_ENV_NAME}
	rm -f .conda-env.${CONDA_ENV_NAME}.build-stamp

.PHONY: check
check: pycodestyle pydocstyle pylint

.PHONY: pycodestyle
pycodestyle: conda-env .pycodestyle.${CONDA_ENV_NAME}.build-stamp

.pycodestyle.${CONDA_ENV_NAME}.build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
.pycodestyle.${CONDA_ENV_NAME}.build-stamp: $(PYTHON_FILES) setup.cfg
	# Check for decent pep8 code style with pycodestyle.
	# Note: if this fails, try using autopep8 to fix it.
	conda run -n ${CONDA_ENV_NAME} pycodestyle $(PYTHON_FILES)
	touch .pycodestyle.${CONDA_ENV_NAME}.build-stamp

.PHONY: pydocstyle
pydocstyle: conda-env .pydocstyle.${CONDA_ENV_NAME}.build-stamp

.pydocstyle.${CONDA_ENV_NAME}.build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
.pydocstyle.${CONDA_ENV_NAME}.build-stamp: $(PYTHON_FILES) setup.cfg
	# Check for decent pep8 doc style with pydocstyle.
	conda run -n ${CONDA_ENV_NAME} pydocstyle $(PYTHON_FILES)
	touch .pydocstyle.${CONDA_ENV_NAME}.build-stamp

.PHONY: pylint
pylint: conda-env .pylint.${CONDA_ENV_NAME}.build-stamp

.pylint.${CONDA_ENV_NAME}.build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
.pylint.${CONDA_ENV_NAME}.build-stamp: $(PYTHON_FILES) .pylintrc
	conda run -n ${CONDA_ENV_NAME} pylint -j0 $(PYTHON_FILES)
	touch .pylint.${CONDA_ENV_NAME}.build-stamp

.PHONY: test
test: pytest

.PHONY: pytest
pytest: conda-env .pytest.${CONDA_ENV_NAME}.build-stamp

.pytest.${CONDA_ENV_NAME}.build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
.pytest.${CONDA_ENV_NAME}.build-stamp: $(PYTHON_FILES) pytest.ini
	#conda run -n ${CONDA_ENV_NAME} pytest -n auto --cov=mlos_core --cov-report=xml mlos_core/ mlos_bench/
	conda run -n ${CONDA_ENV_NAME} pytest --cov --cov-report=xml mlos_core/ mlos_bench/ --junitxml=junit/test-results.xml
	touch .pytest.${CONDA_ENV_NAME}.build-stamp

.PHONY: dist
dist: bdist_wheel

.PHONY: bdist_wheel
bdist_wheel: conda-env mlos_core/dist/mlos_core-*-py3-none-any.whl mlos_bench/dist/mlos_bench-*-py3-none-any.whl

mlos_core/dist/mlos_core-*-py3-none-any.whl: .conda-env.${CONDA_ENV_NAME}.build-stamp
mlos_core/dist/mlos_core-*-py3-none-any.whl: mlos_core/setup.py $(MLOS_CORE_PYTHON_FILES)
	rm -f mlos_core/dist/mlos_core-*-py3-none-any.whl \
	    && cd mlos_core/ \
	    && conda run -n ${CONDA_ENV_NAME} python3 setup.py bdist_wheel \
	    && cd .. \
	    && ls mlos_core/dist/mlos_core-*-py3-none-any.whl

mlos_bench/dist/mlos_bench-*-py3-none-any.whl: .conda-env.${CONDA_ENV_NAME}.build-stamp
mlos_bench/dist/mlos_bench-*-py3-none-any.whl: mlos_bench/setup.py $(MLOS_BENCH_PYTHON_FILES)
	rm -f mlos_bench/dist/mlos_bench-*-py3-none-any.whl \
	    && cd mlos_bench/ \
	    && conda run -n ${CONDA_ENV_NAME} python3 setup.py bdist_wheel \
	    && cd .. \
	    && ls mlos_bench/dist/mlos_bench-*-py3-none-any.whl

.PHONY: dist-test-env-clean
dist-test-env-clean:
	# Remove any existing mlos-dist-test environment so we can start clean.
	conda env remove -y ${CONDA_INFO_LEVEL} -n mlos-dist-test-$(PYTHON_VERSION) 2>/dev/null || true
	rm -f .dist-test-env.$(PYTHON_VERSION).build-stamp

.PHONY: dist-test-env
dist-test-env: dist .dist-test-env.$(PYTHON_VERSION).build-stamp

.dist-test-env.$(PYTHON_VERSION).build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
# Use the same version of python as the one we used to build the wheels.
.dist-test-env.$(PYTHON_VERSION).build-stamp: PYTHON_VERS_REQ=$(shell conda list -n ${CONDA_ENV_NAME} | egrep '^python\s+' | sed -r -e 's/^python\s+//' | cut -d' ' -f1)
.dist-test-env.$(PYTHON_VERSION).build-stamp: mlos_core/dist/mlos_core-*-py3-none-any.whl mlos_bench/dist/mlos_bench-*-py3-none-any.whl
	# Check to make sure we only have a single wheel version availble.
	# Else, run `make dist-clean` to remove prior versions.
	ls mlos_core/dist/mlos_core-*-py3-none-any.whl | wc -l | grep -q -x 1
	ls mlos_bench/dist/mlos_bench-*-py3-none-any.whl | wc -l | grep -q -x 1
	# Symlink them to make the install step easier.
	rm -rf mlos_core/dist/tmp && mkdir -p mlos_core/dist/tmp
	cd mlos_core/dist/tmp && ln -s ../mlos_core-*-py3-none-any.whl mlos_core-latest-py3-none-any.whl
	rm -rf mlos_bench/dist/tmp && mkdir -p mlos_bench/dist/tmp
	cd mlos_bench/dist/tmp && ln -s ../mlos_bench-*-py3-none-any.whl mlos_bench-latest-py3-none-any.whl
	# Create a clean test environment for checking the wheel files.
	$(MAKE) dist-test-env-clean
	conda create -y ${CONDA_INFO_LEVEL} -n mlos-dist-test-$(PYTHON_VERSION) python=$(PYTHON_VERS_REQ)
	conda run -n mlos-dist-test-$(PYTHON_VERSION) pip install pytest pytest-timeout pytest-forked pytest-xdist
	# Test a clean install of the mlos_core wheel.
	conda run -n mlos-dist-test-$(PYTHON_VERSION) pip install "mlos_core/dist/tmp/mlos_core-latest-py3-none-any.whl[full]"
	# Test a clean install of the mlos_bench wheel.
	conda run -n mlos-dist-test-$(PYTHON_VERSION) pip install "mlos_bench/dist/tmp/mlos_bench-latest-py3-none-any.whl[full]"
	touch .dist-test-env.$(PYTHON_VERSION).build-stamp

.PHONY: dist-test
#dist-test: dist-clean
dist-test: dist-test-env .dist-test.$(PYTHON_VERSION).build-stamp

# Unnecessary if we invoke it as "python3 -m pytest ..."
.dist-test.$(PYTHON_VERSION).build-stamp: $(PYTHON_FILES) .dist-test-env.$(PYTHON_VERSION).build-stamp
	# Make sure we're using the packages from the wheel.
	# Note: this will pick up the local directory and change the output if we're using PYTHONPATH=.
	conda run -n mlos-dist-test-$(PYTHON_VERSION) pip list --verbose | grep mlos-core | grep ' pip'
	conda run -n mlos-dist-test-$(PYTHON_VERSION) pip list --verbose | grep mlos-bench | grep ' pip'
	# Run a simple test that uses the mlos_core wheel (full tests can be checked with `make test`).
	conda run -n mlos-dist-test-$(PYTHON_VERSION) python3 -m pytest mlos_core/mlos_core/spaces/tests/spaces_test.py
	# Run a simple test that uses the mlos_bench wheel (full tests can be checked with `make test`).
	conda run -n mlos-dist-test-$(PYTHON_VERSION) python3 -m pytest mlos_bench/mlos_bench/environment/azure/tests/azure_services_test.py
	touch .dist-test.$(PYTHON_VERSION).build-stamp

dist-test-clean: dist-test-env-clean
	rm -f .dist-test-env.$(PYTHON_VERSION).build-stamp

.doc-prereqs.${CONDA_ENV_NAME}.build-stamp: .conda-env.${CONDA_ENV_NAME}.build-stamp
.doc-prereqs.${CONDA_ENV_NAME}.build-stamp: doc/requirements.txt
	conda run -n ${CONDA_ENV_NAME} pip install -U -r doc/requirements.txt
	touch .doc-prereqs.${CONDA_ENV_NAME}.build-stamp

.PHONY: doc-prereqs
doc-prereqs: .doc-prereqs.${CONDA_ENV_NAME}.build-stamp

.PHONY: clean-doc-env
clean-doc-env:
	rm -f .doc-prereqs.build-stamp
	rm -f .doc-prereqs.${CONDA_ENV_NAME}.build-stamp

.PHONY: doc
doc: conda-env doc-prereqs clean-doc
	cd doc/ && conda run -n ${CONDA_ENV_NAME} sphinx-apidoc -f -e -M -o source/api/mlos_core/ ../mlos_core/ ../mlos_*/setup.py
	cd doc/ && conda run -n ${CONDA_ENV_NAME} sphinx-apidoc -f -e -M -o source/api/mlos_bench/ ../mlos_bench/ ../mlos_*/setup.py
	conda run -n ${CONDA_ENV_NAME} make -j -C doc/ html
	test -s doc/build/html/index.html
	test -s doc/build/html/generated/mlos_core.optimizers.BaseOptimizer.html
	test -s doc/build/html/generated/mlos_bench.environment.Environment.html
	test -s doc/build/html/api/mlos_core/mlos_core.html
	test -s doc/build/html/api/mlos_bench/mlos_bench.html
	cp doc/staticwebapp.config.json doc/build/html/

.PHONY: clean-doc
clean-doc:
	rm -rf doc/build/ doc/global/ doc/source/api/ doc/source/generated

.PHONY: clean-check
clean-check:
	rm -f .pylint.build-stamp
	rm -f .pylint.${CONDA_ENV_NAME}.build-stamp
	rm -f .pycodestyle.build-stamp
	rm -f .pycodestyle.${CONDA_ENV_NAME}.build-stamp

.PHONY: clean-test
clean-test:
	rm -f .pytest.build-stamp
	rm -f .pytest.${CONDA_ENV_NAME}.build-stamp
	rm -rf .pytest_cache/
	rm -f coverage.xml .coverage
	rm -rf junit/test-results.xml

.PHONY: dist-clean
dist-clean:
	rm -rf build dist
	rm -rf mlos_core/build mlos_core/dist
	rm -rf mlos_bench/build mlos_bench/dist

.PHONY: clean
clean: clean-check clean-test dist-clean clean-doc clean-doc-env dist-test-clean
	rm -f .conda-env.build-stamp .conda-env.*.build-stamp
	rm -rf mlos_core.egg-info
	rm -rf mlos_core/mlos_core.egg-info
	rm -rf mlos_bench.egg-info
	rm -rf mlos_bench/mlos_bench.egg-info
	rm -rf __pycache__
	find . -type d -name __pycache__ -print0 | xargs -t -r -0 rm -rf
	find . -type f -name '*.pyc' -print0 | xargs -t -r -0 rm -f
