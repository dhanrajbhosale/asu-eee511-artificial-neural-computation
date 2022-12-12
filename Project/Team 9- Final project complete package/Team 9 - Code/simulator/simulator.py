# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:25:40 2022

@author: Sumant Kulkarni
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

    
def draw(n, show=True):
    plt.imshow(n[0],cmap=plt.cm.binary)
    plt.axis('off')
    if show:
        plt.show()

def load_image(img_path, show=True):
    img_file = open(os.path.join(os.getcwd(), img_path), 'rb')
    decoded_data = img_file.read()
    img_data = tf.image.decode_jpeg(decoded_data, channels=3)
    img_data = np.array(img_data).reshape(1, img_data.shape[0], img_data.shape[1], img_data.shape[2])
    draw(img_data, show)
    return img_data
    
def generate_embeddings(img_path, model):
    print("\nThe Test image:")
    img_data = load_image(img_path)
    print("\nGenerating embedding for the Test image:")
    return model.predict(img_data)

def create_embeddings():
    all_embeddings = []
    for i in range(1,31):
        all_embeddings.append(generate_embeddings(
            os.path.join(Path(os.getcwd()).parent, "Datasets\Dog", "d"+str(i)+".jpeg"),model))
        np.save("allembeddings", all_embeddings)
    for i in range(1,31):
        all_embeddings.append(generate_embeddings(
            os.path.join(Path(os.getcwd()).parent, "Datasets\Food", "f"+str(i)+".jpeg"),model))
        np.save("allembeddings", all_embeddings)
    for i in range(1,31):
        all_embeddings.append(generate_embeddings(
            os.path.join(Path(os.getcwd()).parent, "Datasets\Garment", "g"+str(i)+".jpeg"),model))
        np.save("allembeddings", all_embeddings)
    for i in range(1,31):
        all_embeddings.append(generate_embeddings(
            os.path.join(Path(os.getcwd()).parent, "Datasets\Vehicle", "v"+str(i)+".jpeg"),model))
        np.save("allembeddings", all_embeddings)
    for i in range(1,31):
        all_embeddings.append(generate_embeddings(
            os.path.join(Path(os.getcwd()).parent, "Datasets\Location", "l"+str(i)+".jpeg"),model))
        np.save("allembeddings", all_embeddings)

def draw_embedding(index):
    if (index<=30):
        load_image(os.path.join(Path(os.getcwd()).parent, "Datasets\Dog", "d"+str(index)+".jpeg"), show=False)
    elif (index<=60):
        load_image(os.path.join(Path(os.getcwd()).parent, "Datasets\Food", "f"+str(index-30)+".jpeg"), show=False)
    elif (index<=90):
        load_image(os.path.join(Path(os.getcwd()).parent, "Datasets\Garment", "g"+str(index-60)+".jpeg"), show=False)
    elif (index<=120):
        load_image(os.path.join(Path(os.getcwd()).parent, "Datasets\Vehicle", "v"+str(index-90)+".jpeg"), show=False)
    elif (index<=150):
        load_image(os.path.join(Path(os.getcwd()).parent, "Datasets\Location", "l"+str(index-120)+".jpeg"), show=False)


def simulation(category):
    test_image_embedding = generate_embeddings(
                os.path.join(Path(os.getcwd()).parent, "Datasets", "test", category+".jpeg"),model)

    print("\nComparing dataset images with Test image:")
    diff_list = []
    for i, embedding in enumerate(all_embeddings):
        diff_list.append({'index':i, 'diff': np.linalg.norm(test_image_embedding - embedding)})
    diff_list.sort(key = lambda e : e['diff'])
    
    # Show Top 5 images
    print("\nTop 5 Matches are:")
    fig = plt.figure(figsize=(10, 10))
    # setting values to rows and column variables
    rows = 1
    columns = 5
    for i in range(5):
        # Adds a subplot at the current position
        fig.add_subplot(rows, columns, i+1)
        draw_embedding(diff_list[i]['index']+1)    
    plt.show()
    
    # Show all images
    print("\nAll images sorted on embedding similarity:")
    fig = plt.figure(figsize=(15, 10))
    # setting values to rows and column variables
    rows = 10
    columns = 15
    for i in range(150):
        # Adds a subplot at the current position
        fig.add_subplot(rows, columns, i+1)
        draw_embedding(diff_list[i]['index']+1)    
    
#Reduce verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)        
#Load the trained model
print("Loading the trained model:\n")
model = tf.keras.models.load_model('.\submission')
print(model.summary())

all_embeddings = np.load("allembeddings.npy").tolist()
#create_embeddings()
    
simulation("tower")
