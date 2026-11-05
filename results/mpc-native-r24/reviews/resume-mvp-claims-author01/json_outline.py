import json, sys
def walk(x, path, depth, out):
    if isinstance(x, dict):
        for k in x:
            walk(x[k], path + "/" + str(k), depth + 1, out)
    elif isinstance(x, list):
        if len(x) <= 12 and all(not isinstance(v, (dict, list)) for v in x):
            out.append(f"{path} = {json.dumps(x)[:300]}")
        else:
            out.append(f"{path} = list[{len(x)}]")
            for i, v in enumerate(x[:40]):
                walk(v, f"{path}[{i}]", depth + 1, out)
    else:
        s = json.dumps(x)
        out.append(f"{path} = {s[:300]}")
for p in sys.argv[1:]:
    print("=====", p)
    out = []
    walk(json.load(open(p, encoding="utf-8")), "", 0, out)
    print("\n".join(out))
