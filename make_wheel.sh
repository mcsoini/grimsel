rm Grimsel.egg-info dist/ build/ -r
python3 setup.py sdist bdist_wheel

case $1 in 
    -t|--test) echo "Testing"; python3 -m twine upload --repository-url https://test.pypi.org/legacy/ ./dist/*;;
    *) python3 -m twine upload ./dist/*;;
esac
