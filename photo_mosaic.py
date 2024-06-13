import os
from PIL import Image
import numpy as np
import json
import scipy
import heapq

from rich.progress import Progress, TextColumn


class Config:
    def __init__(self):
        self.input_path = 'input.png'
        self.output_path = 'output.png'
        self.tiles_path = 'images'
        self.tile_size = 16
        self.target_tile_size = 16
        self.source_image_alpha = 0.25
        self.brightness_weight = 1.0
        self.color_weight = 1.0
        self.gradient_weight = 0.25
        self.candidates_num = 10
        self.uniqueness_coefficient = 2.0

    def from_json(self, j):
        for key, value in j.items():
            setattr(self, key, value)
    
    def to_json(self):
        return {key: getattr(self, key) for key in vars(self)}


config = Config()


class Tile:
    def __init__(self, array: np.ndarray):
        self.array = array
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        mono = np.dot(array, [299, 587, 114]) / 1000
        self.brightness = np.mean(np.array(mono)) / 255.0
        self.color = np.mean(self.array, axis=(0, 1)) / 255.0
        i22 = np.array(Image.fromarray(mono.astype(np.uint8)).resize((2, 2)))
        grad_x = np.mean(np.abs(i22[1:, :] - i22[:-1, :])) / 255.0
        grad_y = np.mean(np.abs(i22[:, 1:] - i22[:, :-1])) / 255.0
        self.gradient = np.array([grad_x, grad_y])
        self.features = np.concatenate([
            config.brightness_weight * np.array([self.brightness]), 
            config.color_weight * self.color,
            config.gradient_weight * self.gradient
            ])


def load_tiles():
    tiles = []
    with Progress(
        *Progress.get_default_columns(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}")
    ) as progress:
        for filename in progress.track(os.listdir(config.tiles_path), description='Loading tiles'):
            image_path = os.path.join(config.tiles_path, filename)
            try:
                image = Image.open(image_path)
                image = image.convert('RGB')
                image = image.resize((config.target_tile_size, config.target_tile_size))
                tiles.append(Tile(np.array(image)))
            except:
                pass
    print(f'Loaded {len(tiles)} tiles')
    return tiles


def export_tile_stats(tiles):
    def get_stats(name, values):
        return { name: {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }}
    stats = {}
    stats.update(get_stats('brightness', [tile.brightness for tile in tiles]))
    stats.update(get_stats('color-r', [tile.color[0] for tile in tiles]))
    stats.update(get_stats('color-g', [tile.color[1] for tile in tiles]))
    stats.update(get_stats('color-b', [tile.color[2] for tile in tiles]))
    stats.update(get_stats('gradient-x', [tile.gradient[0] for tile in tiles]))
    stats.update(get_stats('gradient-y', [tile.gradient[1] for tile in tiles]))
    with open('tile_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)


def create_mosaic(tiles):
    try:
        input_image = Image.open(config.input_path)
        input_image = input_image.convert('RGB')
        input_image = input_image.resize((
            input_image.width // config.tile_size * config.tile_size, 
            input_image.height // config.tile_size * config.tile_size))
    except:
        print(f'Failed to load input image "{config.input_path}"')
        return None
    
    tiles_features = np.array([tile.features for tile in tiles]) # (N, 5)

    image_tiles = []
    for y in range(input_image.height // config.tile_size):
        for x in range(input_image.width // config.tile_size):
            image_tile = Tile(np.array(input_image.crop((
                x * config.tile_size, y * config.tile_size, 
                (x + 1) * config.tile_size, (y + 1) * config.tile_size))))
            image_tiles.append(image_tile)
    
    image_features = np.array([tile.features for tile in image_tiles]) # (M, 5)

    D = scipy.spatial.distance.cdist(image_features, tiles_features, 'euclidean') # (M, N)
    closest_tile_indices = np.argsort(D, axis=1) # (M, N)
    K = min(max(1, config.candidates_num), closest_tile_indices.shape[1])
    closest_tile_indices = closest_tile_indices[:, :K] # (M, K)

    tile_uniqueness = max(1, int(len(image_tiles) / len(tiles) * config.uniqueness_coefficient))
    tile_usages = np.zeros(len(tiles), dtype=np.int32)
    heap = [(D[i, closest_tile_indices[i, 0]], (i, 0)) for i in range(len(image_tiles))]
    heapq.heapify(heap)
    
    output_image = np.zeros((
        input_image.height // config.tile_size * config.target_tile_size, 
        input_image.width // config.tile_size * config.target_tile_size, 3), dtype=np.uint8)

    with Progress(
        *Progress.get_default_columns(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
    ) as progress:
        total_tiles = (input_image.height // config.tile_size) * (input_image.width // config.tile_size)
        task = progress.add_task('Creating mosaic', total=total_tiles)

        while len(heap) > 0:
            _, (i, j) = heapq.heappop(heap)
            tile_index = closest_tile_indices[i, j]
            if tile_usages[tile_index] >= tile_uniqueness:
                if j + 1 < K:
                    heapq.heappush(heap, (D[i, closest_tile_indices[i, j + 1]], (i, j + 1)))
                    continue
                else:
                    p = [1.0 / (D[i, tile_index] + 1e-6) for tile_index in closest_tile_indices[i]]
                    p = np.array(p) / np.sum(p)
                    tile_index = np.random.choice(closest_tile_indices[i], p=p)
            tile_usages[tile_index] += 1

            x = i % (input_image.width // config.tile_size)
            y = i // (input_image.width // config.tile_size)
            input_image_tile = np.array(Image.fromarray(image_tiles[i].array).resize((config.target_tile_size, config.target_tile_size)))
            alpha = min(1.0, max(0.0, config.source_image_alpha))
            output_image[y * config.target_tile_size:(y + 1) * config.target_tile_size, 
                         x * config.target_tile_size:(x + 1) * config.target_tile_size] = \
                tiles[tile_index].array * (1.0 - alpha) + input_image_tile * alpha
            progress.update(task, advance=1)

    return Image.fromarray(output_image)


def main():
    tiles = load_tiles()
    export_tile_stats(tiles)
    mosaic = create_mosaic(tiles)
    if mosaic is not None:
        print(f'Saving mosaic to "{config.output_path}"')
        mosaic.save(config.output_path)
        mosaic.show()


if __name__ == '__main__':
    config = Config()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Config file path, default: "config.json"')
    parser.add_argument('--input', help=f'Input image path, default: "{config.input_path}"')
    parser.add_argument('--output', help=f'Output image path, default: "{config.output_path}"')
    parser.add_argument('--tiles-path', help=f'Path to tiles, default: "{config.tiles_path}"')
    parser.add_argument('--tile-size', type=int, help=f'Tile size to split input image, default: {config.tile_size}')
    parser.add_argument('--target-tile-size', type=int, help=f'Target tile size for each tile in output image, default: {config.target_tile_size}')
    parser.add_argument('--source-image-alpha', type=float, help=f'Alpha value for source image, default: {config.source_image_alpha}')
    parser.add_argument('--brightness-weight', type=float, help=f'Weight for brightness difference, default: {config.brightness_weight}')
    parser.add_argument('--color-weight', type=float, help=f'Weight for color difference, default: {config.color_weight}')
    parser.add_argument('--gradient-weight', type=float, help=f'Weight for gradient difference, default: {config.gradient_weight}')
    parser.add_argument('--candidates-num', type=int, help=f'Number of candidates for each tile, default: {config.candidates_num}')
    parser.add_argument('--uniqueness-coefficient', type=float, help=f'Coefficient for uniqueness of tiles, default: {config.uniqueness_coefficient}')
    args = parser.parse_args()

    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.from_json(json.load(f))
        except:
            print(f'Failed to load config file "{args.config}", using default config.')
    
    for key, value in vars(args).items():
        if key != 'config' and value:
            setattr(config, key, value)

    print('Config:', json.dumps(config.to_json(), indent=4))

    main()