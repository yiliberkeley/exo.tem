import os
import PCV

from PCV.clustering import hcluster
from PIL import Image
from matplotlib.pyplot import *
from numpy import *

# create a list of images
path = 'data/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
# extract feature vector (8 bins per color channel)
features = zeros([len(imlist), 8])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 1), 8, range=[(0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)

# visualize clusters with some (arbitrary) threshold
clusters = tree.extract_clusters(0.23 * tree.distance)
# plot images for clusters with more than 3 elements
for c in clusters:
    elements = c.get_cluster_elements()
    nbr_elements = len(elements)
    if nbr_elements > 3:
        figure()
        for p in range(minimum(nbr_elements,20)):
            subplot(4, 5, p + 1)
            im = array(Image.open(imlist[elements[p]]))
            imshow(im)
            axis('off')
show()

hcluster.draw_dendrogram(tree,imlist,filename='exo_tem.pdf')
