#!/bin/bash

mkdir -p build/chapters/

pdflatex -output-directory=build report.tex
bibtex build/report.aux
pdflatex -output-directory=build report.tex
pdflatex -output-directory=build report.tex

cp build/report.pdf .
