#!/bin/sh

jupyter nbconvert ../notebooks/doc_running_grimsel.ipynb --to rst 
mv ../notebooks/doc_running_grimsel.rst ./source/

jupyter nbconvert ../notebooks/doc_introductory_example.ipynb --to rst 
mv ../notebooks/doc_introductory_example.rst ./source/

make html

