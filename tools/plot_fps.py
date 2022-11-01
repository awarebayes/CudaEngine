import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

running_fps = defaultdict(list)
lastmin = None

with open("logs/avg_fps_per_nthreads_orig.csv", "r") as f:
    for line in f:
        nthreads, fps = line.split(",")
        nthreads = int(float(nthreads))
        if lastmin is None:
            lastmin = nthreads
        if nthreads == lastmin:
            running_fps[int(float(nthreads))].append(float(fps))
        elif nthreads < lastmin:
            lastmin = nthreads

#for nthreads, fps in running_fps.items():
    #if (len(fps) > 100):
        #plt.plot(fps[100:], label=str(nthreads))

#plt.xlabel("Номер измерения")
#plt.ylabel("Среднее количество кадров в секунду")
#plt.title("Среднее количество кадров в секунду в зависимости для n потоков")
#plt.legend()

labels = []
averages = []
for nthreads, fps in running_fps.items():
    if (len(fps) > 50):
        labels.append(int(nthreads))
        averages.append(sum(fps[50:]) / len(fps[50:]))


zipped_lists = zip(labels, averages)
sorted_pairs = sorted(zipped_lists, key=lambda x: x[0])
tuples = zip(*sorted_pairs)
labels, averages = [list(tuple) for tuple in  tuples]

plt.bar(labels, averages)
plt.plot(labels, averages, label="Среднее количество кадров в секунду", color="red")
plt.title("Среднее количество кадров в секунду в зависимости для n потоков")
plt.xlabel("Количество потоков")
plt.ylabel("Среднее количество кадров в секунду")
plt.show()

table = pd.DataFrame({'nthreads': labels, 'average': averages})
print(table.to_latex(index=False))