print("TEST.PY")
import torch
import segsemdata
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
dfc2015test = segsemdata.makeISPRStestVAIHINGEN(
    datasetpath="/data/ISPRS_VAIHINGEN", normalize=False, color=True
)
nbclasses = len(dfc2015test.getcolors())

print("load model")
net = torch.load("build/model.pth")
net = net.to(device)
net.eval()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("do test")
cm = np.zeros((nbclasses, nbclasses), dtype=int)
allproducts = dfc2015test.apply(net, 128, 64, pathout="build/")
for i in range(len(allproducts)):
    cm += confusion_matrix(
        allproducts[i][0].flatten(), allproducts[i][1].flatten(), list(range(nbclasses))
    )
print("accuracy=", np.sum(cm.diagonal()) / (np.sum(cm) + 1))
print(cm)
