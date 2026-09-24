from scalar_node import AutoNode

a = AutoNode(2)
b = AutoNode(3)
c = a * b
d = c + a
assert c.value == 6
assert d.value == 8
print("passed")
