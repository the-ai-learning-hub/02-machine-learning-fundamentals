import sys, nbformat
sys.stdout.reconfigure(encoding="utf-8")
nb=nbformat.read(sys.argv[1],4)
for c in nb.cells:
    if c.cell_type=="code":
        for o in c.get("outputs",[]):
            if o.get("output_type")=="stream": print(o.text[:3000])
            elif o.get("output_type")=="execute_result": print(o["data"]["text/plain"])
            elif o.get("output_type")=="error": print("ERR",o["ename"],o["evalue"])
