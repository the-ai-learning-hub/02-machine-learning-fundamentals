#!/bin/bash
# uso: run_uv.sh 03 <caminho do notebook>  (usa o .venv do projeto com uv)
cd C:/Users/diogo/Desktop/02-machine-learning-fundamentals
uv run python _planning/builders/build_$1.py || exit 1
s=$(date +%s)
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$2" 2>&1 | grep -v -E "Proactor|_get_loop|without encryption" | tail -25
echo "exit nbconvert: ${PIPESTATUS[0]}"; echo "SEGUNDOS: $(( $(date +%s)-s ))"
uv run python -c "import nbformat,sys;nb=nbformat.read(sys.argv[1],4);nbformat.validate(nb);import re;bad=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error' or (o.get('output_type')=='stream' and o.get('name')=='stderr')];print('valid | erros/stderr:',len(bad));s=open(sys.argv[1],encoding='utf-8').read();print('caminhos locais:',len(re.findall(r'C:\\\\Users|C:/Users|diogo',s)))" "$2"
ls -la "$2" | awk '{print $5}'
