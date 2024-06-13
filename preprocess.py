import os
from PIL import Image
from pillow_heif import register_heif_opener
import face_recognition

register_heif_opener()


TARGET_SIZE = 256

def resize_images(input_path, output_path, face_image):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # prepare target face encoding
    target_face_encoding = None
    if face_image and os.path.exists(face_image):
        face_image = face_recognition.load_image_file(face_image)
        face_encodings = face_recognition.face_encodings(face_image)
        if not face_encodings:
            print("No face found in face image")
        target_face_encoding = face_encodings[0]

    # resize images
    for filename in os.listdir(input_path):
        image_path = os.path.join(input_path, filename)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(output_path, f'{name}.png')

        try:
            with Image.open(image_path) as image:
                width, height = image.size

                # skip small images
                if width < TARGET_SIZE or height < TARGET_SIZE:
                    print(f"Skipping {image_path} of size {width}x{height}")
                    continue
                
                # resize large images
                if width > TARGET_SIZE and height > TARGET_SIZE:
                    ratio = max(TARGET_SIZE / width, TARGET_SIZE / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    image = image.resize((new_width, new_height))
                image.save(save_path)
                
                # check if image contains target face
                target_face_location = (image.width // 2, image.height // 2)
                if target_face_encoding is not None:
                    image_fr = face_recognition.load_image_file(save_path)
                    face_locations = face_recognition.face_locations(image_fr)
                    face_encodings = face_recognition.face_encodings(image_fr, face_locations)
                    for i, face_encoding in enumerate(face_encodings):
                        match = face_recognition.compare_faces([target_face_encoding], face_encoding)
                        if match[0]:
                            top, right, bottom, left = face_locations[i]
                            center_x = (left + right) // 2
                            center_y = (top + bottom) // 2
                            target_face_location = (center_x, center_y)
                            break
                
                # crop image so that face is in the center
                left = max(0, min(target_face_location[0] - TARGET_SIZE // 2, image.width - TARGET_SIZE))
                top = max(0, min(target_face_location[1] - TARGET_SIZE // 2, image.height - TARGET_SIZE))
                right = left + TARGET_SIZE
                bottom = top + TARGET_SIZE
                image = image.crop((left, top, right, bottom))
                print(f"Cropped {image_path} of size {width}x{height} with {left},{top},{right},{bottom}")
                image.save(save_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='source_images')
    parser.add_argument('--output-path', type=str, default='images')
    parser.add_argument('--face-image', type=str, default='face.png')
    args = parser.parse_args()

    resize_images(args.input_path, args.output_path, args.face_image)