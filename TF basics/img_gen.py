import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

print(os.getcwd())
print(os.path())
# Directory with our training horse pictures
train_horse_dir = os.path.join('./images/train/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('./images/train/horse-or-human/humans')

train_dir = os.path.join('./images/train/horse-or-human')


train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size =  (300, 300),
    batch_size = 128,
    class_mode= 'binary'
)
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])


model.compile(
    optimizer = 'adagrad',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
    )


