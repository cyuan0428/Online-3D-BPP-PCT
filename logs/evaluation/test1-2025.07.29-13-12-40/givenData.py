# container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.
# item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.

# container_size = [10,10,10]
container_size = [80,90,50]


item_size_set = [
    [20, 44, 7],  # type 1
    [20, 88, 7],  # type 2
    [40, 44, 7],  # type 3
    [20, 22, 7],  # type 4
]
# lower = 1
# higher = 5
# resolution = 1
# item_size_set = []
# for i in range(lower, higher + 1):
#     for j in range(lower, higher + 1):
#         for k in range(lower, higher + 1):
#                 item_size_set.append((i * resolution, j * resolution , k *  resolution))

# If you want to sample item sizes from a uniform distribution in continuous domain,
# type --sample-from-distribution in your command line.
