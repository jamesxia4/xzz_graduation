import numpy as np
pre=np.load("y_sh.npy")[0:100000]
gt=np.load("y_trues.npy")[0:100000]
from matplotlib import pyplot as  plt
plt.plot(pre)
plt.plot(gt)
plt.show()