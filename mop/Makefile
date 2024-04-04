
SHELL:= /bin/bash

# BEGIN: FINE TUNE
FILES:= draft
DUST_EXT:= {aux,bbl,blg,dvi,log,nav,out,Rout,snm,synctex.gz,toc,vrb}
# to keep .tex & .log files from 'knitr::knit()', uncomment next line
#.PRECIOUS: %.tex 
SHAREDCOMM:= awk '/begin{shared-comm.tex/ {p=1}; p; /end{shared-comm.tex/ {p=0}' ms.Rnw > shared-comm.tex
PREP:=$(SHAREDCOMM)
# END: FINE TUNE 

PDFL_OPTS:= -output-format pdf

RSCRIPT = Rscript --vanilla

.PHONY: default
default:
	for f in $(FILES); do (make $$f.pdf); done

.PHONY: help
help:
	@echo "Type 'make' for the default recipe (see steps below), i.e.: (i) clean, (ii) make all pdf files, and (iii) dust"
	@echo "Type 'make clean' to remove files named after tex or Rnw files and with extension pdf or of DUST_EXT (see 'make dust' below)"
	@echo "Type 'make filename.pdf' to make a pdf from filename.tex or filename.Rnw (by knitting it into filename.tex)"
	@echo "Type 'make dust' to remove files named after tex or Rnw files and with extension one of (DUST_EXT, i.e.):" $(DUST_EXT)

%.pdf: %.tex
	pdflatex $(PDFL_OPTS) $*; bibtex $*; pdflatex $(PDFL_OPTS) $*; pdflatex $(PDFL_OPTS) $*

%.tex: %.Rnw
	Rscript --no-save --no-restore --no-init-file -e "knitr::knit(\"$*.Rnw\",quiet=TRUE)" > $*.Rout 2>&1

%.R: %.Rnw
	$(RSCRIPT) -e "library(knitr); purl(\"$*.Rnw\")"

.PHONY:clean dust
clean: dust
	for f in $(FILES); do (rm -f $(basename $$f).pdf $(basename $$f).dvi $(basename $$f).html); done;\
	$(foreach f, $(wildcard *.Rnw), rm -f $(basename $f).tex $(basename $f).R)
	rm -f cache/figs/*.*

dust:
	for f in $(FILES); do (rm -f $(basename $$f).$(DUST_EXT)); done
