content-mit-latex2edx-demo
==========================

latex2edx demo course

This repository contains an example edX course generated via latex2edx (https://github.com/mitocw/latex2edx).

Live Example Site
=================

https://edge.edx.org/courses/MITx/MIT.latex2edx/2014_Spring/about

LaTeX Source Files
==================

The main tex files are in the "src" directory.

See these latex files for specific examples of latex code with latex2edx macros:

* https://github.com/mitocw/content-mit-latex2edx-demo/blob/master/src/basic.tex

* https://github.com/mitocw/content-mit-latex2edx-demo/blob/master/src/advanced.tex

To compile this course, check out this repository, go to src, and run "make"

Contents
========

This is a sample course generated from a latex source file, compiled
into edX XML using latex2edx. Content includes examples of basic
problems, text, and video modules, as well as advanced problems using
custom graders written in python, and graphical input via javascript.

Examples included:

* Numerical response with inline labels
* Custom response problem with two inputs
* Formula response problem checking mathematical equations
* Matrix expression equality testing (for math expressions with nonabelian operators
* Quantum mechanics ket input
* Adaptive hints using "hint_fn" scripts
* Drag and drop problem created via latex2dnd

See the latex2edx distribution and the latex2dnd package (https://github.com/mitocw/latex2dnd) for more information.
