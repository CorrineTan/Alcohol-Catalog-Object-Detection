import keras

from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np

model = VGG19(weights='imagenet', include_top=False)

def get_feature_from_NN(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    features = model.predict(img_data)
    return features

import glob

# open the output index file for writing
output = open('index_nn.csv', "w")

# use glob to grab the image paths and loop over them
for ext in ('*.gif', '*.png', '*.jpg','*.jpeg'):
    for imagePath in glob.glob('Users/tengluntan/Dropbox/product-images' + ext):
        print(imagePath)
        # extract the image ID (i.e. the unique filename) from the image# path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        #image = cv2.imread(imagePath)
        features_nn=get_feature_from_NN(img_path)

        # write the features to file
        features = features_nn[0].reshape(1,7*7*512)
        features = [str(f) for f in features.tolist()]
        output.write("%s,%s\n" % (imageID, ",".join(features)))
print("finish populating features per image")
# close the index file
output.close()

import numpy as np
import csv

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit = 50):
        results = {}

        with open(self.indexPath) as f:
            reader = csv.reader(f)

            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                results[row[0]] = d
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def chi2_distance(self, histA, histB, eps = 1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        return d

#from search_engine.searcher import Searcher
template_features= get_feature_from_NN('crops/93_1_roi.jpeg', model)
# perform the search
searcher = Searcher('index_nn.csv')
results = searcher.search(template_features)

import matplotlib.pyplot as plt
# display the query
im_template=plt.imread('crops/93_1_roi.jpeg')
plt.imshow(im_template)
plt.title("image submited for query")
plt.show()
print("----"*25)
resultimgs=[]
# loop over the results
for (score, resultID) in results:
    # load the result image and display it
    print(score,resultID)
    # fix window path  \ and convert to /
    result=plt.imread(resultID.replace('\\','/')) 
    resultimgs.append(result)
    plt.imshow(result)
    
    plt.title("search result image ID : "+resultID)
    plt.show()
