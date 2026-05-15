# Define the directories containing projects
LIBS_DIRS := $(wildcard libs/*)

# Default target
.PHONY: all
all: lint format lock test

# Install dependencies for all projects
.PHONY: install
install:
	@printf 'Creating virtual environment...\n'
	@uv venv
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/pyproject.toml ]; then \
			@printf 'Installing dependencies for %s\n' "$$dir"; \
			uv pip install -e $$dir; \
		fi; \
	done

# Lint all projects
.PHONY: lint
lint:
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/Makefile ]; then \
			@printf 'Running lint in %s\n' "$$dir"; \
			$(MAKE) -C $$dir lint; \
		fi; \
	done

# Format all projects
.PHONY: format
format:
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/Makefile ]; then \
			@printf 'Running format in %s\n' "$$dir"; \
			$(MAKE) -C $$dir format; \
		fi; \
	done

# Lock all projects
.PHONY: lock
lock:
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/Makefile ]; then \
			@printf 'Running lock in %s\n' "$$dir"; \
			(cd $$dir && uv lock); \
		fi; \
	done

# Lock all projects and upgrade dependencies
.PHONY: lock-upgrade
lock-upgrade:
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/Makefile ]; then \
			@printf 'Running lock-upgrade in %s\n' "$$dir"; \
			(cd $$dir && uv lock --upgrade); \
		fi; \
	done

# Test all projects
.PHONY: test
test:
	@for dir in $(LIBS_DIRS); do \
		if [ -f $$dir/Makefile ]; then \
			@printf 'Running test in %s\n' "$$dir"; \
			$(MAKE) -C $$dir test; \
		fi; \
	done