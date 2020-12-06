from Njh import Njh
from time import time

x = time()
Njh('./corpus.txt', window_size=[1,1])
print(f'{time() - x} seconds')