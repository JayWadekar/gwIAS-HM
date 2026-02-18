SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html clean

html:
	python3 scripts/generate_api_tree.py
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)/*
