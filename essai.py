import numbers

print(isinstance(1.0, numbers.Number))


print(isinstance(complex(1.0, 0.0), numbers.Real))


print(isinstance(-1, numbers.Real))
