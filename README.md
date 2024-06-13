# photo-mosaic
A python script to generate photo mosaic.

## Usage

### Requirements

Use `pip install` to install the following pakages:
```
pillow pillow_heif face_recognition numpy scipy rich
```

### Pre-process

Pre-process will resize all tile images to a smaller squared size (256x256).
1. Put all tile images into a folder (e.g. "source_images").
2. (Optional) Put an image containing a face (e.g. "face.png") so that the resizing will try to focus on it.
3. Run `preprocess.py`.

### Create Mosaic Image

1. Prepare an input image (e.g. "input.png").
2. (Optional) Use a "config.json" such as [config.json.example](config.json.example) to specify parameters.
3. Run `photo_mosaic.py`.
