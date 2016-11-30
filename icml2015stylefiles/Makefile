PAPER = example_paper
TEX = $(wildcard *.tex)
BIB = *.bib
FIGS = $(wildcard *.pdf *.eps *.png)

.PHONY: all clean once

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS)
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

# JZ: fast make when there's no bib update -- usage: make once
once:	$(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

