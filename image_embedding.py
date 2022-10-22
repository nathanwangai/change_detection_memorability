import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import spatial
from skimage.transform import resize

# ----- start of Image class -----

""" 
The Image class automatically computes and stores the top 5 predictions and selected layer embeddings for an input image.
"model" is a tensorflow model that returns selected intermediate layer activations as the output.
"""
class Image: 
    def __init__(self, img_path, model):
        self.model = model # "predictions" layer must be included in the submodel
        self.input_dim = self.model.input.shape[1]
        self.img = self.load_img(img_path)
        self.top5 = self.predict_top5()
        self.embeddings = self.get_embeddings()

    # formats image according to network specifications
    def load_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_dim, self.input_dim, 3))
        return img
    
    # returns list of tuples for top 5 class predictions and the corresponding probability
    def predict_top5(self):
        predictions = self.model.predict(np.expand_dims(self.img, axis=0))[-1]
        top5_raw = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=5)[0]
        top5 = []

        for pred in top5_raw: 
            pred_class = pred[1]
            pred_prob = pred[2]*100
            top5.append((pred_class, pred_prob))

        return top5

    # store intermediate layer activations of model
    def get_embeddings(self): 
        all_embeds = self.model(np.expand_dims(self.img, axis=0))
        unravel_embeds = []

        for embed in all_embeds:
            unravel_embeds.append(tf.keras.layers.Flatten()(embed))

        return unravel_embeds

    # prints contents of the "top5" attribute
    def print_top5(self):
        for pred in self.top5:
            print(pred[0] + ": %.2f" % pred[1])
        print("")

    # displays images of Image objects side by side
    def show_next_to(self, img2): 
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        plt.imshow(self.img/255)
        plt.subplot(1,2,2)
        plt.imshow(img2.img/255)
        plt.show()

# ----- end of Image class -----

# ----- start of helper functions -----

# store all layer names of "model" to a list
def get_all_layer_names(model):
    layer_names = []

    for layer in model.layers: # layer attributes: name, output, activation
        layer_names.append(layer.name)

    return layer_names

# define submodel of "model" with outputs as the activations of "layers"
def get_submodel(layers, model):
    outputs = []

    for layer in layers:
        extracted_layer = model.get_layer(layer)
        outputs.append(extracted_layer.output)
        print(layer, extracted_layer.output.shape)
    print("")

    submodel = tf.keras.models.Model(inputs=model.input, outputs=outputs)
    return submodel

# compute the cosine similarity between two vectors
def cosine_sim(vec1, vec2):
    return (1 - spatial.distance.cosine(vec1, vec2))*100

# compute cosine similarity of corresponding layer activations between two images
def similarity_vs_layer(img1, img2):
    similarity_scores = []

    for i in range(7):
        similarity = cosine_sim(img1.embeddings[i], img2.embeddings[i])
        similarity_scores.append(similarity)

    return similarity_scores

# plot "similarity_scores" outputted from "similarity_vs_layer()"
def plot_similarities(feature_layers, similarity_scores):
    plt.plot(feature_layers, similarity_scores)
    plt.title("Similarity Between Image Pairs vs. Model Layer", fontsize=16, fontweight="bold")
    plt.xlabel("Layer ID", fontsize=14, fontweight="bold")
    plt.xticks(rotation = 45)
    plt.ylabel("Cosine Similarity x100", fontsize=14, fontweight="bold")
    
# ----- end of helper functions -----

if __name__ == "__main__":
	pass