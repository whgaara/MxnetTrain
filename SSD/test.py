import numpy as np
ttt = np.random.randint(0, 10, (3, 2, 2))
print(ttt)

print('---------------------\n')

ttt1 = ttt.transpose((1, 2, 0))
print(ttt1)
print(ttt1.flatten())