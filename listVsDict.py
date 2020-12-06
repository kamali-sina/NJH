import time
import numpy as np

num = 10000000
r = range(num)
lis = list(r)
dick = {i:i for i in r}
b= 1
stime = time.time()
sequence = np.random.choice(num, size=num, replace=True)
for i in sequence:
    b = lis[i]
print(f'list took {time.time() - stime} seconds')

stime = time.time()
for i in sequence:
    b = dick[i]
print(f'dict took {time.time() - stime} seconds')

stime = time.time()
for i in sequence:
    b = dick.get(i, 'fuck')
print(f'dict and default val took {time.time() - stime} seconds')