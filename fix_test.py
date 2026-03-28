with open("tests/test_logistic_coverage.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("from g.models import LogisticErrorCode, LogisticMethod"): continue
    new_lines.append(line)

new_lines.insert(2, "from g.compute.logistic import LogisticErrorCode, LogisticMethod\n")

with open("tests/test_logistic_coverage.py", "w") as f:
    f.write("".join(new_lines))
