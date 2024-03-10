.PHONY: format check types

all: format check types

format:
	tox -e format

check:
	tox -e lints,types

types:
	tox -e types
