import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import openai
import requests
from typing import Tuple

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

def generate_dalle_image(animal_name: str, width: int, height: int) -> Tuple[bytes, np.ndarray]:
    """Generate an image using DALL-E API based on animal name."""
    try:
        # Get OpenAI API key from environment
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create prompt based on animal name - make it more descriptive and safe
        animal_desc = animal_name.replace('_', ' ')
        prompt = f"A cute cartoon illustration of a {animal_desc} in a whimsical art style, colorful and friendly"
        
        # Generate image using DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Download the generated image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        # Open image (no need to resize since it's already 1024x1024)
        image = Image.open(io.BytesIO(image_response.content))
        image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.uint8)
        
        return image_array.tobytes(), image_array
        
    except Exception as e:
        print(f"DALL-E generation failed for {animal_name}: {e}")
        print(f"Retrying with simpler prompt...")
        try:
            # Retry with a simpler, safer prompt
            simple_prompt = f"A {animal_name.split('_')[-1]} animal cartoon"
            response = client.images.generate(
                model="dall-e-3",
                prompt=simple_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            image = Image.open(io.BytesIO(image_response.content))
            image = image.convert('RGB')
            image_array = np.array(image, dtype=np.uint8)
            
            return image_array.tobytes(), image_array
            
        except Exception as e2:
            print(f"Retry also failed: {e2}")
            print("Falling back to random image generation...")
            return generate_fallback_image(width, height)

def generate_funny_animal_names(count: int) -> list:
    """Generate funny animal names using OpenAI API."""
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Generate {count} funny, creative animal names in the style of Ubuntu version names (adjective_animal format).  Make them humorous and imaginative but family-friendly. Examples: "giggling_giraffe", "dancing_dolphin", "sneezing_sloth".  Return only the names, one per line, with underscores between words, no numbering or extra text."""
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8
        )
        
        names = [name.strip() for name in response.choices[0].message.content.strip().split('\n') if name.strip()]
        return names[:count]  # Ensure we don't get more than requested
        
    except Exception as e:
        print(f"Failed to generate funny names: {e}")
        print("Using fallback names...")
        return [
            "giggling_giraffe", "dancing_dolphin", "sneezing_sloth", "bouncing_bear", "singing_seal",
            "tickled_tiger", "laughing_llama", "jumping_jaguar", "winking_whale", "chuckling_cheetah"
        ][:count]

def generate_fallback_image(width, height):
    """Generate a random RGB image as fallback."""
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return image_array.tobytes(), image_array

def save_image_as_jpeg(image_array, filename):
    """Save numpy image array as JPEG file."""
    image = Image.fromarray(image_array, 'RGB')
    image.save(filename, format='JPEG', quality=95)

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
    width, height = 1024, 1024
    num_samples = 5
    
    # Generate funny animal names using OpenAI
    print("Generating funny animal names...")
    names = generate_funny_animal_names(num_samples)
    
    # Find the longest name to determine padding
    max_length = max(len(name) for name in names)
    
    # Create output directory for images
    os.makedirs("generated_images", exist_ok=True)
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in range(num_samples):
            # Generate DALL-E image
            print(f"Generating image {i+1}/{num_samples} for {names[i]}...")
            image_bytes, image_array = generate_dalle_image(names[i], width, height)
            # Pad label to consistent length
            label = names[i].ljust(max_length)
            
            # Save image as JPEG
            image_filename = f"generated_images/{names[i]}.jpg"
            save_image_as_jpeg(image_array, image_filename)
            
            # Create tf.train.Example
            tf_example = create_tf_example(label, image_bytes, width, height)
            
            # Write to TFRecord
            writer.write(tf_example.SerializeToString())
            
            print(f"Created example {i+1}/{num_samples}: '{label}' -> {image_filename}")
    
    print(f"\nTFRecord file '{output_file}' created successfully with {num_samples} examples.")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    print(f"Label length: {max_length} characters")

if __name__ == "__main__":
    main()
