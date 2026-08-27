.PHONY: install scan xcode

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r python/requirements.txt
	$(VENV)/bin/pip install -e python

xcode:
	open macos/StrataIdent.xcodeproj

extract:
	$(PY) -m strata_ident extract $(VIDEO) --out frames --fps 2

enroll:
	$(PY) -m strata_ident enroll --name "$(NAME)" --gallery gallery.json $(PHOTOS)

scan:
	$(PY) -m strata_ident scan $(PATH) --gallery gallery.json --json report.json
