import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file")
parser.add_argument("--output-file")
args = parser.parse_args()

def _ctc_postprocess(tokens):
    _toks = tokens.split()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != "<blank>"]
    hyp = " ".join(hyp)
    return hyp

with open(args.input_file) as f:
    data = f.read().splitlines()

output = []
for line in data:
    output.append(_ctc_postprocess(line))

with open(args.output_file, "w") as f:
    f.write("\n".join(output))