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

### Parameters

- `input_path`: the input image that will be composed of many tiles.
- `output_path`: the output mosaic image.
- `tiles_path`: the folder that contains all pre-processed tile images.
- `tile_size`: the input image will be splitted into blocks of size `tile_size`, and each block will be replaced with an image from `tile_path`.
- `target_tile_size`: the output image tile size. Tile images will be resized to this size. The output image width will be `input image width` / `tile_size` * `target_tile_size`.
- `source_image_alpha`: if greater than zero, source image will be overlaid on the output image, making the output image more similar to the input image in details.
- Weights: used to select the best tile image to replace the block of input image.
  - `brightness_weight`: Weight for brightness. The larger, the output image more likely to look like a mono-color image.
  - `color_weight`: Weight for color. Increase this if `tile_size` is small, so that the output image matches the input image in color.
  - `gradient_weight`: Weight for gradient. Increase this if `tile_size` is large so that blocks will have more details in shape.
- Uniqueness parameters: Controls the algorithm that selects the best tile image to replace the input image blocks.
  - `candidates_num`: Number of candidates (tile images) for each block to choose from.
  - `uniqueness_coefficient`: The larger, the output image more likely to have many blocks that share the same tile image. The algorithm will try not to use one tile image for more than `K` times, where `K = blocks / number of tile images * uniqueness_coefficient`. But if all candidate images are used more than `K` times, the algorithm will randomly select from all candidates weighted by block-tile similarity.
