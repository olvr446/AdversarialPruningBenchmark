# from utils.utils import load_model, model_key_maker, load_distance
# from utils.plots import plot_sec_curve
# from utils.test import test_model_aa, test_model_hofmn
# from taxonomy.utils import load_ap_taxonomy

# ap = "FlyingBird_Chen2021Sparsity"
# arch = "resnet18"
# ds = "CIFAR10"
# struct = "weights"  # or filters, channels
# sr = "90"

# # get a unique model key
# model_key = model_key_maker(ap_method=ap,
#                             architecture=arch,
#                             dataset=ds,
#                             structure=struct,
#                             sparsity_rate=sr)

# # when get_distances, the distances computed with the best config from HOFMN is returned in addition to the model
# model = load_model(model_key=model_key)
# distances = load_distance(model_key=model_key)

# # test the model
# rob_acc_aa = test_model_aa(model=model, ds=ds, data_dir='AdversarialPruningBenchmark', device='cpu')
# rob_acc_hofmn = test_model_hofmn(model, model_key=model_key, ds='CIFAR10', data_dir='/Users/giorgiopiras/Desktop/PhD/PythonProjects/AdversarialPruningBenchmark/AdversarialPruningBenchmark', device='cpu', loss='DLR', optimizer='SGD', scheduler='CALR', get_distances=False)

# # plot security curve
# plot_sec_curve(distances, names=[model_key], title=model_key+'DLR_SGD_CALR', save=True)

# print(f'Model {model_key} AA robust accuracy: {rob_acc_aa}')
# print(f'Model {model_key} HO-FMN robust accuracy: {rob_acc_hofmn}')
# print(f'Within the taxonomy, here are the AP entries: {load_ap_taxonomy(ap)}')

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./HOFMN'))
from utils.utils import load_model, model_key_maker, load_distance
from utils.plots import plot_sec_curve
from utils.test import test_model_aa, test_model_hofmn
from taxonomy.utils import load_ap_taxonomy
import torch


ap = "HYDRA_Sehwag2020Hydra"
arch = "resnet18"
ds = "CIFAR10"
struct = "weights"  # or filters, channels
sr = "95"

# get a unique model key
model_key = model_key_maker(ap_method=ap,
                            architecture=arch,
                            dataset=ds,
                            structure=struct,
                            sparsity_rate=sr)

# get model and distances given their unique key
model = load_model(model_key=model_key)
distances = load_distance(model_key=model_key)

# test the model 
# clean_acc, rob_acc_aa = test_model_aa(model, ds=ds, data_dir='my_datadir/CIFAR10', device='cpu')
# rob_acc_hofmn, _ = test_model_hofmn(model, model_key=model_key, dataset=ds, data_dir='my_datadir/CIFAR10', device='cpu', loss='DLR', optimizer='SGD', scheduler='CALR', get_distances=False)
model.eval().to("cpu")
with torch.no_grad():
    output = model(torch.rand(1, 3, 32, 32))
print(output.shape)
# plot security curve (you can compare even more models together)
names = [model_key]  # add a name for each model to appear in the legend
plot_sec_curve(distances, names, title=model_key+'DLR_SGD_CALR', save=False)

# print(f'Model {model_key} clean accuracy: {clean_acc}')
# print(f'Model {model_key} AA robust accuracy: {rob_acc_aa}')
# print(f'Model {model_key} HOFMN robust accuracy: {rob_acc_hofmn}')
print(f'Within the taxonomy, here are the AP entries: {load_ap_taxonomy(ap)}')