import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10,6)
sns.set()


results = pd.read_pickle('./results_multiple.pickle')
means = {}
for v in set(results.step):
    rel = results[results['step']==v]
    for key, feature in rel.items():
        if key == 'step':
            continue
        if key not in means.keys():
            means[key] = []

        means[key].append(feature.mean())
print(means)

fig = plt.figure()
for i in range(1,5,1):
    plt.subplot(2,2,i)
    k = list(means.keys())[i]
    vals = means[k]
    plt.plot(np.arange(0, 501, 100), vals, '-', label=k)
    plt.xlabel('step')
    plt.legend()
plt.tight_layout()
plt.suptitle("Multi-class Classification: Means of indexes in 10 times", y=0.05)
fig.savefig("./code/colored_mnist/result_multiple.jpg")

