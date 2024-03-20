import numpy as np
import matplotlib.pyplot as plt

a = np.load("a.npy")
b = np.load("b.npy")
x = np.load("x.npy")
x1 = np.load("x1.npy")

print(a.shape)
print(b.shape)
print(x.shape)
print(x1.shape)

a = (a.reshape((a.shape[0], 56, 56, a.shape[2]))*256).astype(np.uint8)
b = (b.reshape((b.shape[0], 56, 56))*256).astype(np.uint8)
x = (x.reshape((x.shape[0], 56, 56))*256).astype(np.uint8)
x1 = (x1.reshape((x1.shape[0], 56, 56))*256).astype(np.uint8)
# x1 = (x1.reshape((x1.shape[0], 56, 56, x1.shape[2]))*256).astype(np.uint8)

print(f"a : {a.shape} b : {b.shape}")

print(a[0, :, :, 0])

for i in range(5):
    plt.subplot(1, 5, i+1)
    k = np.random.randint(0, a.shape[3])
    plt.imshow(a[0, :, :, k])
    plt.savefig('a.png')

# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     k = np.random.randint(0, a.shape[3])
plt.close()
print(np.unique(b))
plt.imshow(b[0, :, :])
plt.savefig('b.png')
plt.close()
print(np.unique(x))
plt.imshow(x[0, :, :])
plt.savefig('x.png')
plt.close()
print(np.unique(x1))
plt.imshow(x1[0, :, :])
plt.savefig('x.png')

# print(b[0].reshape((56, 56)))
# plt.imshow(b[0].reshape((56, 56)))
# plt.savefig('b.png')