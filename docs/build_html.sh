#!/bin/sh

jupyter nbconvert ../notebooks/doc_running_grimsel.ipynb --to rst 
mv ../notebooks/doc_running_grimsel.rst ./source/

make html

