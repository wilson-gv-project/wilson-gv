#!/usr/bin/env python
import numpy as np

def nm_pattern(nmodes: int, numberofindices: int):
    """
    Generate a pattern of indices for a tensor of nmodes.
    :param nmodes:
    :return:
    """
    # Define the shape of the tensor
    shape = tuple([nmodes for _ in range(numberofindices)])
    # print('>>>', shape)
    # Create the tensor using broadcasting
    tensor = np.indices(shape).transpose(*[i for i in range(1, numberofindices + 1)] + [0])

    flattened_tensor = tensor.reshape(-1, numberofindices)

    # Create an empty list to store the selected indices
    selected_indices = []

    # Iterate through the flattened tensor and select indices of the form (a, a, b, b)
    for indices in flattened_tensor:
        if indices[0] == indices[2] and indices[1] == indices[3]:
            selected_indices.append(indices)

    selected_indices_array = np.array(selected_indices)
    return selected_indices_array


# import numpy as np
#
# # Define the shape of the tensor
# shape = (3, 3, 3, 3)
#
# # Create the tensor using broadcasting
# tensor = np.indices(shape).transpose(1, 2, 3, 4, 0)
#
# # Print the tensor
# print(tensor)
# print(tensor.shape)
#
# flattened_tensor = tensor.reshape(-1, 4)
#
# # Create an empty list to store the selected indices
# selected_indices = []
#
# # Iterate through the flattened tensor and select indices of the form (a, a, b, b)
# for indices in flattened_tensor:
#     if indices[0] == indices[2] and indices[1] == indices[3]:
#         selected_indices.append(indices)
#
# # Convert the list to a numpy array
# selected_indices_array = np.array(selected_indices)


a = np.arange(9).reshape(3, 3)
b = np.arange(10, 19).reshape(3, 3)

a = np.arange(1,5).reshape(-1, 1)
# print((1/a)*(1/a).T)
print(a*a.T)

b = np.arange(6, 10).reshape(4, 1)

# Reshape w to (n, 1, 1) where n is the number of elements in w
# w = np.arange(1,5).reshape(-1, 1, 1)
w = np.arange(1,5)
w2 = np.arange(6, 20, 4)

wext = w[:, np.newaxis]
w2ext = w2[np.newaxis, :]

w2ext2 = w2[:, np.newaxis]
wext2 = w[np.newaxis, :]
print('-------------------')
print('w2ext\n', w2ext, '\nwext\n', wext)
print('-------------------')
print('w2ext2\n', w2ext2, '\nwext\n',wext2)
print('-------------------')
print(w2ext-wext)
print(w2ext2-wext2)

print('-------------------')
# Create a tensor of shape (6, 6, 1, 23, 7)
tensor = np.random.rand(2, 2, 1, 3, 3)
print(tensor)
print(tensor.shape)
# Extend the third dimension (index 2) from 1 to 6
extended_tensor = np.tile(tensor, (1, 1, 2, 1, 1))

# Check the new shape of the tensor
print("New shape:", extended_tensor.shape)
print(extended_tensor)
quit()
# Using np.einsum to create a 3D product
# 'i,j,k->ijk' indicates that the output should have dimensions corresponding to each input
# i.e., it multiplies each element in w with every element in w and w itself, creating a 3D array
pref_Tab_3D = np.einsum('i,j,k->ijk', w, w, w)
pref_Tab_2D = np.einsum('i,j->ij', w, w)
# pref_Tab_3D = np.einsum('i11,j11,k11->ijk', w, w, w)

print(pref_Tab_3D)
print(pref_Tab_2D)

# print(a)
# print(b)
# print('---------')
# mat2 = np.hstack([a] * len(a))
# print(mat2)
mat2 = np.vstack([a.T] * len(a))
# print(mat2)
# quit()
# print(a.T)
# print(a[:, np.newaxis]*a.T)
# print(np.outer(a, a.T))


mat3 = np.vstack([b] * 3)
print(mat3)
# print(mat2*mat3)

# quit()

selected_indices_array = nm_pattern(5, 4)

print(selected_indices_array)

data_tensor = np.random.rand(5, 5, 5, 5)
# print(data_tensor)

tot = 0.
for inx in selected_indices_array:
    print(data_tensor[inx[0], inx[1], inx[2], inx[3]])
    tot += data_tensor[inx[0], inx[1], inx[2], inx[3]]

print(tot)

# quit()
# Calculate the sum of values from data_tensor with indices from selected_indices_array
sum_of_values = np.sum(data_tensor[selected_indices_array[:, 0], selected_indices_array[:, 1], selected_indices_array[:, 2], selected_indices_array[:, 3]], axis=0)

# Print the sum
print(sum_of_values)

print('\n---------------------------------------------------------------------')
nmodes = 3
totB = np.zeros((nmodes, nmodes, nmodes))
totB[2, 1, :] = 1
# print(totB)


# Define the range of random integers
low = 1    # Inclusive lower bound of the random integers
high = 10  # Exclusive upper bound of the random integers

# Generate random integers for w_ab and w_a
w_ab = np.random.randint(low, high, size=(6, 6))  # w_(a+b) tensor of shape (6, 6)
w_a = np.random.randint(low, high, size=(6))      # w_a tensor of shape (6,)


# Reshape w_a to (6, 1) to enable broadcasting along the correct dimension
w_a_reshaped = w_a.reshape(6, 1)
print('----------------------')
print(w_ab)
print(w_a_reshaped)
# Perform the subtraction
result = w_ab - w_a_reshaped
print(result)

quit()

print('\n---------------------------------------------------------------------')

# Assuming tensor1 is your (6, 3) array, tensor2 is your (6, 3, 3) array, and tensor3 is your (6, 6, 3) array
tensor1 = np.random.rand(6, 3)
tensor2 = np.random.rand(6, 3, 3)
tensor3 = np.random.rand(6, 6, 3)
tensor4 = np.random.rand(6, 6, 3, 3)

# Reshape tensors to match the final shape
# mu_Q, alpha_Q, mu_QQ
tensor1_reshaped = tensor1.reshape(6, 1, 1, 1, 1, 3, 1, 1)
tensor2_reshaped = tensor2.reshape(1, 6, 1, 1, 3, 1, 1, 3)
tensor3_reshaped = tensor3.reshape(1, 1, 6, 6, 1, 1, 3, 1)
# tensor4_reshaped = tensor4.reshape(1, 1, 6, 6, 1, 1, 3, 3)

# mu_Q, alpha_QQ, mu_Q
tensor1_reshaped = tensor1.reshape(6, 1, 1, 1, 1, 3, 1, 1)
tensor4_reshaped = tensor4.reshape(1, 1, 6, 6, 3, 1, 1, 3)
tensor1_reshaped2 =tensor1.reshape(1, 6, 1, 1, 1, 1, 3, 1)


# Combine tensors
result = tensor1_reshaped * tensor2_reshaped * tensor3_reshaped
print(result)
print(result.shape)
print(np.prod(result.shape))

print('\n---------------------------------------------------------------------')

o = result[:, :, :, :, 0, 0, 1, 1]
print(o)
print(o.shape)

