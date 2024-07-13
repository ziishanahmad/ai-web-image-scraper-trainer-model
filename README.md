

# AI Web Image Scraper and Trainer Model

This project consists of a web image scraper that downloads images from a specified website, saves them along with their alt tags, and trains a Convolutional Neural Network (CNN) model to classify these images based on their alt tags. It also includes functionality to predict and display the label of a new image using the trained model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Downloading Images](#downloading-images)
  - [Training the Model](#training-the-model)
  - [Predicting Image Labels](#predicting-image-labels)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ziishanahmad/ai-web-image-scraper-trainer-model.git
   cd ai-web-image-scraper-trainer-model
   ```

2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook scrapping_image_model.ipynb
   ```

## Usage

### Downloading Images

To download images from a specified website and save them along with their alt tags, run the first few cells of the notebook:

1. **Import necessary libraries and define the scraping function:**
   ```python
   import requests
   from bs4 import BeautifulSoup
   import os
   from urllib.parse import urljoin, urlparse
   from collections import deque
   import pandas as pd

   def download_images_from_website(url, output_dir, csv_path, max_depth=3):
       ...
   ```

2. **Set parameters and run the scraping function:**
   ```python
   website_url = 'https://www.apple.com/'
   output_parent_directory = '/content/drive/My Drive/datasets/applescrapper1'
   csv_file_path = '/content/drive/My Drive/datasets/ddata1.csv'
   max_depth = 3

   if not os.path.exists(output_parent_directory):
       os.makedirs(output_parent_directory)

   download_images_from_website(website_url, output_parent_directory, csv_file_path, max_depth)
   ```

### Training the Model

After downloading the images, proceed to train the CNN model using the preprocessed data:

1. **Mount Google Drive and check for GPU:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import tensorflow as tf

   if not tf.test.gpu_device_name():
       raise SystemError('GPU device not found')
   print('Found GPU at: {}'.format(tf.test.gpu_device_name()))
   ```

2. **Load the CSV data and preprocess it:**
   ```python
   import pandas as pd
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   csv_file_path = '/content/drive/My Drive/datasets/ddata1.csv'
   df = pd.read_csv(csv_file_path)

   datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

   train_generator = datagen.flow_from_dataframe(
       dataframe=df,
       directory=None,
       x_col='image_path',
       y_col='alt_tag',
       target_size=(224, 224),
       batch_size=32,
       class_mode='categorical',
       subset='training'
   )

   val_generator = datagen.flow_from_dataframe(
       dataframe=df,
       directory=None,
       x_col='image_path',
       y_col='alt_tag',
       target_size=(224, 224),
       batch_size=32,
       class_mode='categorical',
       subset='validation'
   )
   ```

3. **Build and train the CNN model:**
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
   from tensorflow.keras.callbacks import EarlyStopping
   import json

   num_classes = len(train_generator.class_indices)

   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
       MaxPooling2D((2, 2)),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D((2, 2)),
       Flatten(),
       Dense(64, activation='relu'),
       Dense(num_classes, activation='softmax')
   ])

   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

   early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

   history = model.fit(train_generator, epochs=5, validation_data=val_generator, callbacks=[early_stopping])

   # Save the model and class indices
   model.save('/content/drive/My Drive/datasets/model.h5')

   class_indices = train_generator.class_indices
   with open('/content/drive/My Drive/datasets/class_indices.json', 'w') as f:
       json.dump(class_indices, f)
   ```

### Predicting Image Labels

To predict the label of a new image using the trained model:

1. **Load the pre-trained model and class indices:**
   ```python
   from tensorflow.keras.models import load_model
   import json

   model_path = '/content/drive/My Drive/datasets/model.h5'
   model = load_model(model_path)

   class_indices_path = '/content/drive/My Drive/datasets/class_indices.json'
   with open(class_indices_path, 'r') as f:
       class_indices = json.load(f)
   labels = {v: k for k, v in class_indices.items()}
   ```

2. **Define the prediction function and display the image with its predicted label:**
   ```python
   from tensorflow.keras.preprocessing import image
   import matplotlib.pyplot as plt
   import numpy as np

   def predict_image_label(image_path, model, labels):
       img = image.load_img(image_path, target_size=(224, 224))
       img_array = image.img_to_array(img)
       img_array = np.expand_dims(img_array, axis=0) / 255.0

       predictions = model.predict(img_array)
       predicted_class = np.argmax(predictions, axis=1)[0]

       return img, labels[predicted_class]

   new_image_path = '/content/drive/My Drive/datasets/applescrapper1/hero_iphone_pro_max__bsan8nevcgty_large.jpg'
   img, predicted_label = predict_image_label(new_image_path, model, labels)

   img_array = np.array(img) / 255.0
   plt.imshow(img_array)
   plt.title(f"Predicted Label: {predicted_label}")
   plt.axis('off')
   plt.show()
   ```

## Project Structure

```
.
├── scrapping_image_model.ipynb  # Jupyter notebook with the entire workflow
├── requirements.txt             # List of required libraries
└── README.md                    # This README file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

