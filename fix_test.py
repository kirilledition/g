with open("tests/test_logistic_coverage.py", "r") as f:
    content = f.read()

content = content.replace("standard_coefficients = jnp.array(\n        [\n            [0.1, 0.2, 0.3],\n            [0.4, 0.5, 0.6],\n        ]\n    )", "standard_coefficients = jnp.zeros((64, 3))")

with open("tests/test_logistic_coverage.py", "w") as f:
    f.write(content)
