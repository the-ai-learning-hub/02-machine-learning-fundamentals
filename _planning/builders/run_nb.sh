#!/bin/bash
# uso: run_nb.sh 03 <caminho do notebook>
PY="C:/Users/diogo/AppData/Local/Temp/claude/5120910ceae76249/68e8bf41-4274-4d58-8c46-670b2aeb583c/scratchpad/venv/Scripts/python.exe"
cd C:/Users/diogo/Desktop/02-machine-learning-fundamentals
$PY _planning/builders/build_$1.py || exit 1
s=$(date +%s)
$PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$2" 2>&1 | tail -20
echo "SEGUNDOS: $(( $(date +%s)-s ))"
$PY -c "import nbformat,sys;nbformat.validate(nbformat.read(sys.argv[1],4));print('valid')" "$2"
ls -la "$2"
