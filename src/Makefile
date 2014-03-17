# makefile for 8.421

all:	mitx.xhtml intro_only.xhtml basic_only.xhtml advanced_only.xhtml

mitx.xhtml: mitx.tex 
	latex2edx  mitx.tex
	./RELOAD

intro_only.xhtml: intro.tex 
	latex2edx -m intro_only.tex
	./RELOAD

basic_only.xhtml: basic.tex 
	latex2edx -m basic_only.tex
	./RELOAD

advanced_only.xhtml: advanced.tex 
	latex2edx -m advanced_only.tex
	./RELOAD
