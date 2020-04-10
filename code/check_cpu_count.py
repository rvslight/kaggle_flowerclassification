import multiprocessing
import os
import psutil
total_logical_cpu_count = psutil.cpu_count(logical = True)
print("logical_cpu_count: ",total_logical_cpu_count)
print(os.getcwd())