# Hard-and-Soft-Clustering
### C-mean Soft Clustering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
data = pd.read_csv('ceramic.csv')
data = data[['Na2O', 'Al2O3']]
xpts = data['Na2O']
ypts = data['Al2O3']

fig1, axes1 = plt.subplots(2, 2, figsize=(9, 9))
alldata = np.vstack((xpts, ypts))
fpcs = []
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    
    fpcs.append(fpc)
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.')
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')
        
    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    

fig1.tight_layout()

fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:6], fpcs)
ax2.set_xlabel("Center")
ax2.set_ylabel("FCP")
```

![image](https://user-images.githubusercontent.com/58222828/173663029-1b722d0b-9e3d-4c11-8481-400ab13ed0b0.png)
![image](https://user-images.githubusercontent.com/58222828/173663051-5e1be222-c930-4c71-99d8-bdbe9e72df10.png)

To find the optimal number of clusters in fuzzy clustering or soft clustering, I used Fuzzy Partition Coefficient or FPM. The FPC is defined on the range from 0 to 1, with 1 being best. In particularly, it calculates the all membership degrees of each point in dataset, and shows use the “diffusion degree” of dataset. It is a metric which tells us how cleanly our data is distributed by a certain model.

---

### K-mean Hard Clustering

```python
import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('ceramic.csv')
data = data[['Na2O', 'Al2O3']]
x1 = data['Na2O']
x2 = data['Al2O3']

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,6),locate_elbow=True)
visualizer.fit(data)      
visualizer.show()
```

![image](https://user-images.githubusercontent.com/58222828/173663089-dfa012a6-28fc-4776-98c3-121414e606f8.png)

After fitting the model, result shows that the 3 cluster will be the most optimal number of centers for our dataset.
