import numpy as np
# import os
# y_true=np.load('/Data/zhouhang/anomaly_detection/gcn_mil/y_trues.npy')
# y_pre=np.load('/Data/zhouhang/anomaly_detection/gcn_mil/y_preds.npy')

# print(y_pre.shape)
# print(y_true.shape)
def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()

if __name__ == "__main__":
    for i in range(3):
        write_flow(np.ones((10,10)),("p%d.txt")%i)