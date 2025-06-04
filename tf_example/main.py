import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_list_feature(value):
    """Returns a bytes_list from a list of string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def generate_random_image(width, height):
    """Generate a random RGB image and return as raw bytes."""
    # Generate random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # Return raw bytes (height * width * 3 bytes)
    return image_array.tobytes()

def create_tf_example(label, image_bytes, width, height):
    """Create a tf.train.Example from image data."""
    feature = {
        'label': _bytes_feature(label.encode('utf-8')),
        'image/encoded': _bytes_feature(image_bytes),
        'image/shape': _int64_list_feature([width, height, 3]),  # RGB image has 3 channels
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
    output_file = "sample_images.tfrecord"
    width, height = 640, 480
    num_samples = 10
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in range(num_samples):
            # Generate random image
            image_bytes = generate_random_image(width, height)
            label = f"sample_image_{i}"
            
            # Create tf.train.Example
            tf_example = create_tf_example(label, image_bytes, width, height)
            
            # Write to TFRecord
            writer.write(tf_example.SerializeToString())
            
            print(f"Created example {i+1}/{num_samples}: {label}")
    
    print(f"\nTFRecord file '{output_file}' created successfully with {num_samples} examples.")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
