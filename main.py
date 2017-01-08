import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2

train_dir = r'G:\work\LiLab\DeepEyes\Datasets\d_manatee\sketches'
test_dir = r'G:\work\LiLab\DeepEyes\Datasets\d_manatee\test_set'
ht = 32
wd = 64
num_nbr = 20


def get_sketch(sketch_path):
    sketch = cv2.imread(sketch_path)
    if sketch is not None:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        sketch = cv2.resize(sketch, (wd, ht))
        sketch = (sketch/255.).astype('float32')
        return sketch.flatten()
    else:
        print "Unable to open ", sketch_path
        return np.zeros(ht*wd).astype('float32')
    
    
if __name__ == '__main__':
    #train_sketch_names = ['DU041.tif', 'DU065.jpg','DU232.tif', 'DU310.jpg']
    train_sketch_names = os.listdir(train_dir)
    
    print 'Reading training data..',
    X_train = np.empty([len(train_sketch_names), ht * wd], dtype='float32')
    for idx, sketch_name in enumerate(train_sketch_names):
        X_train[idx] = get_sketch(os.path.join(train_dir, sketch_name))
    print 'Done.'
    
    # Train
    print 'Training started.',
    nbrs = NearestNeighbors(n_neighbors=num_nbr, algorithm='auto').fit(X_train)
    print 'Done.'
    
    #Test
    #test_sketches = ['U1523_B.jpg', 'U1955_B.jpg', 'U2028_B.tif']
    test_sketches = os.listdir(test_dir)
    
    print 'Testing started.',
    X_test = np.empty([len(test_sketches), ht * wd], dtype='float32')
    for idx, test_sketch_name in enumerate(test_sketches):
        X_test[idx] = get_sketch(os.path.join(test_dir, test_sketch_name))

    distances, indices = nbrs.kneighbors(X_test)
    print 'Done.'
    
    for idx, test_sketch_name in enumerate(test_sketches):
        top_matches = [train_sketch_names[index].split('.')[0] for index in indices[idx][0:num_nbr]]
        if test_sketch_name.split('.')[0] in top_matches:
            print test_sketch_name, "Match found at rank: ", top_matches.index(test_sketch_name.split('.')[0])
        else:
            print test_sketch_name, "Match not found in top ", num_nbr
    




    