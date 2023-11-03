
SHELL:= /bin/bash

# BEGIN: FINE TUNE
FILES:= ms
DUST_EXT:= {aux,bbl,blg,dvi,log,nav,out,Rout,snm,synctex.gz,toc,vrb}
# to keep .tex & .log files from 'knitr::knit()', uncomment next line
#.PRECIOUS: %.tex
SHAREDCOMM:= awk '/begin{shared-comm.tex/ {p=1}; p; /end{shared-comm.tex/ {p=0}' ms.Rnw > shared-comm.tex
PREP:=$(SHAREDCOMM)
# END: FINE TUNE

PDFL_OPTS:= -output-format pdf

PDF_FILES:= $(wildcard figure/*.pdf)
TIF_FILES:= $(patsubst %.pdf,%.tif,$(PDF_FILES))

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

clean:
	rm -f *.aux *.bbl *.blg *.Rout *.log *.out *-concordance.tex *.gz

# clean: dust
# 	for f in $(FILES); do (rm -f $(basename $$f).pdf $(basename $$f).dvi $(basename $$f).html); done;\
# 	$(foreach f, $(wildcard *.Rnw), rm -f $(basename $f).tex $(basename $f).R)
# 	rm -f cache/figs/*.*

# dust:
# 	for f in $(FILES); do (rm -f $(basename $$f).$(DUST_EXT)); done

%.tif: %.pdf
	pdftoppm -tiff $*.pdf $*
	Rscript --vanilla removeTrailingNums.R $*-1.tif

tifs: $(TIF_FILES)

submission: ms.tex
	Rscript --vanilla remove_knitrout.R ms.tex
	pdflatex ms-submission.tex
	bibtex ms-submission.aux
	Rscript --vanilla append_bib.R ms-submission.tex
	pdflatex ms-submission.tex
	pdflatex ms-submission.tex
	rm ms-submission.log ms-submission.blg ms-submission.out ms-submission.aux ms-submission.bbl
	if ! [ -d "submission" ]; then mkdir submission; fi
	mv ms-submission.pdf submission/
	mv ms-submission.tex submission/

ArXiv: ms.tex ms.bbl si/si.tex si/calibrateMod3Out.tex si/ReplicateLee20Out.tex
	if ! [ -d "ArXiv" ]; then mkdir ArXiv; fi
	cp ms.tex ArXiv/
	cp ms.bbl ArXiv/
	cp plos2015.bst ArXiv/
# 	cp bib-haiti.bib ArXiv/ms.bib
	cp -r figure/. ArXiv/
	rm -r ArXiv/*.tif
	cp si/si.tex ArXiv/
	cp si/si.bbl ArXiv/
	cp -r si/inputs/ ArXiv/
	cp -r si/figure/ ArXiv/
	cp si/calibrateMod3Out.tex ArXiv/
	cp si/ReplicateLee20Out.tex ArXiv/
	Rscript --vanilla ArXiv.R ArXiv/ms.tex
	rm ArXiv/ms.bbl
	rm ArXiv/si.bbl
	zip -vr ArXiv.zip ArXiv/ -x "*.DS_Store"
