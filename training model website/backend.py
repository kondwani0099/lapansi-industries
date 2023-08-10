# from flask import Flask, render_template, request, jsonify
# import os

# # app = Flask(__name__)

# app = Flask(__name__, static_folder='static')


# UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded files
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/train', methods=['POST'])
# def train():
#     print('data recieved')
#     learning_type = request.form['learning-type']
#     dataset = request.files['dataset']
#     validation = request.files['validation']
#     testing = request.files['testing']

#     # Save uploaded files to the UPLOAD_FOLDER
#     # dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.zip')
#     # dataset.save(dataset_path)

#     # Before saving the file
#     dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
#     dataset.save(dataset_path)


#     # Implement your training process using the uploaded dataset
#     # You can use machine learning libraries like TensorFlow, PyTorch, etc.

#     # Generate example data for the loss-accuracy chart
#     training_data = [0.8, 0.6, 0.4, 0.2, 0.1]
#     validation_data = [0.7, 0.5, 0.3, 0.2, 0.15]

#     results = {
#         'training_data': training_data,
#         'validation_data': validation_data
#     }

#     return jsonify(results)
# @app.route('/submit', methods=['POST'])
# def submit_data():
#     if request.method == 'POST':
#         learning_type = request.form['learning-type']
#         dataset = request.files['dataset']
#         validation = request.files['validation']
#         testing = request.files['testing']

#         # Process and store the data in the database
#         # Implement database insertion code here

#         return "Data submitted successfully"


from flask import Flask, render_template, request, jsonify
import os
import zipfile
from PIL import Image
import shutil
from flask import jsonify
from os.path import join, dirname, realpath

import zipfile
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator
app = Flask(__name__)
UPLOAD_FOLDER= join(dirname(realpath(__file__)), 'uploads/')
# UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ... (Previous code)

import tensorflow as tf
import os
os.chmod('temp/train/Apple', 0o644)

def preprocess_data(uploaded_files):
    processed_data = []
    temp_dir = 'temp'

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    for file in uploaded_files:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

    # Process images from the subfolders
    subfolders = ['train', 'valid', 'test']
    for subfolder in subfolders:
        subfolder_path = os.path.join(temp_dir, subfolder)
        images = []
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image)
            image = tf.image.resize(image, (224, 224))
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
            images.append(image)
        processed_data.append(images)

    shutil.rmtree(temp_dir)

    return processed_data

# ... (Rest of the code)

# def preprocess_data(uploaded_files):
#     processed_data = []
#     temp_dir = 'temp'

#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)
#     os.makedirs(temp_dir)

#     for file in uploaded_files:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)

#     # Process images from the subfolders
#     subfolders = ['train', 'valid', 'test']
#     for subfolder in subfolders:
#         subfolder_path = os.path.join(temp_dir, subfolder)
#         images = []
#         for filename in os.listdir(subfolder_path):
#             image_path = os.path.join(subfolder_path, filename)
#             image = Image.open(image_path)
#             resized_image = image.resize((224, 224))
#             images.append(resized_image)
#         processed_data.append(images)

#     shutil.rmtree(temp_dir)

#     return processed_data
def remove_directory(directory):
    def onerror(func, path, exc_info):
        """
        Error handler for 'shutil.rmtree'.
        If the error is due to an access error (read-only file),
        it attempts to add write permission and then retries.
        """
        import stat
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise

    shutil.rmtree(directory, onerror=onerror)

# def train_model(processed_data):
    # data_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # resnet = models.resnet18(pretrained=True)
    # num_fruit_classes = 5  # Change this based on your dataset

    # resnet.fc = nn.Linear(resnet.fc.in_features, num_fruit_classes)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    # train_data = processed_data[0]
    # train_labels = torch.tensor([0, 1, 2, 3, 4])  # Change labels based on your dataset

    # for epoch in range(10):  # Adjust number of epochs as needed
    #     running_loss = 0.0
    #     for images in train_data:
    #         inputs = torch.stack([data_transforms(image) for image in images])
    #         optimizer.zero_grad()
    #         outputs = resnet(inputs)
    #         loss = criterion(outputs, train_labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_data)}")

    # return resnet



# def preprocess_data(uploaded_files):
#     print('unzipping')
#     processed_data = []
#     temp_dir = 'temp'
#     for file in uploaded_files:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             zip_ref.extractall('temp')  # Extract to a temporary directory

#              # Clean up the temporary directory if it already exists
#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)
#     os.makedirs(temp_dir)

#     for file in uploaded_files:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)

#         image_paths = [os.path.join(temp_dir, filename) for filename in os.listdir(temp_dir)]

#         processed_images = []

#         # image_paths = [os.path.join('temp', filename) for filename in os.listdir('temp')]

#         # processed_images = []
#         for image_path in image_paths:
#             os.chmod(image_path, 0o644)  # Set read permissions for owner, group, and others

#             image = Image.open(image_path)
#             resized_image = image.resize((224, 224))
#             processed_images.append(resized_image)

#         # Add processed data to the list
#         processed_data.append(processed_images)

#     # Clean up the temporary directory
#     shutil.rmtree('temp')

#     return processed_data

# def train_model(processed_data):
#     # Implement your model training logic here
#     # Return the trained model
#     return trained_model
# ... (Previous code)



def train_model(data_directory):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_directory, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_directory, 'valid'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'
    )

    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_fruit_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    return model

# ... (Previous code)



def train_model(data_directory):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_directory, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'  # Use 'sparse' for integer labels
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_directory, 'valid'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'
    )

    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_fruit_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    return model



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    print("Received POST request")

    learning_type = request.form['learning-type']

    # Loop through uploaded files and save them
    uploaded_files = request.files.getlist('file')  # 'file' is the name attribute of your file input fields
    file_paths = []

    for file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_paths.append(file_path)
   

    # Process and prepare the data
    processed_data = preprocess_data(uploaded_files)

    # Train the model
    trained_model = train_model(processed_data)
    data_directory = preprocess_data(uploaded_files)  # You can remove this line if not needed
    trained_model = train_model(data_directory)

    # Process and train the model
    # ...

    # Generate example data for the loss-accuracy chart
    training_data = [0.8, 0.6, 0.4, 0.2, 0.1]
    validation_data = [0.7, 0.5, 0.3, 0.2, 0.15]

    return render_template('results.html', training_data=training_data, validation_data=validation_data)


if __name__ == '__main__':
    app.run(debug=True)
