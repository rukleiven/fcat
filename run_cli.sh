# Script for running tests. The first argument lists which python command to run
# If you wish to use python3 run this script as
# bash run_cli.sh python3. If not given, python will be used
PYCMD=${1-python}

echo "Running using $PYCMD"

echo "Testing main command help"
${PYCMD} scripts/cli.py --help

echo "Test linearize"
${PYCMD} scripts/cli.py linearize --template
${PYCMD} scripts/cli.py linearize --config linearize_template.yml --out model.json

echo "Test linstep"
${PYCMD} scripts/cli.py linstep --mod model.json --out data.csv --step_var=3 --tmax=100.0
rm linearize_template.yml model.json data.csv
