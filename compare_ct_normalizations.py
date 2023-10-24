import argparse
import numpy as np
from pathlib import Path

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, WindowedCTNormalization


class Normalizer:
    def __init__(self):
        self.normalizations = {
            'CTNormalization': {
                'class': CTNormalization,
                'arguments': {
                    'intensityproperties': {
                        "max": 3603.951904296875,
                        "mean": -3.0302115922610047,
                        "median": 52.470924377441406,
                        "min": -1464.62890625,
                        "percentile_00_5": -868.7071533203125,
                        "percentile_99_5": 610.099609375,
                        "std": 232.17899606521533
                    }
                }
            },
            'CTLung': {
                'class': WindowedCTNormalization,
                'arguments': {
                    'window_level': -500,
                    'window_width': 1400,
                    'intensityproperties': {} # required as dict, not used
                }
            },
            'CTAbdomen': {
                'class': WindowedCTNormalization,
                'arguments': {
                    'window_level': 40,
                    'window_width': 350,
                    'intensityproperties': {} # required as dict, not used
                }
            }
        }

    def normalize_volumes(self, path_to_volumes, path_to_output):
        paths_to_volumes = list(Path(path_to_volumes).glob('*.nii.gz'))
        reader_writer = SimpleITKIO()
        for path_to_volume in paths_to_volumes:
            print(f"study: {path_to_volume.name}")
            volume, properties = reader_writer.read_images([path_to_volume])
            for name, norm in self.normalizations.items():
                print(f"normalization scheme: {name}")
                normalizer = norm['class'](**norm['arguments'])
                normalized = normalizer.run(volume)
                print(f"min: {np.min(normalized)}")
                print(f"max: {np.max(normalized)}")
                print(f"min: {np.min(normalized)}")
                print(f"mean: {np.mean(normalized)}")
                print(f"std: {np.std(normalized)}")
                if name == 'CTNormalization':
                    normalized = (normalized - np.min(normalized)) \
                        / (np.max(normalized) - np.min(normalized))
                grayscale = np.around(normalized * 255)
                output_filename = Path(path_to_output) / f"{path_to_volume.name.split('.nii.gz')[0]}_{name}.nii.gz"
                reader_writer.write_seg(grayscale[0, ...], output_filename, properties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Generate normalized CT studies to compare different
        normalization strategies""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_studies',
        type=str,
        help="""Path to the directory containing input studies in NIfTI
        format."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save the output normalized studies."
    )
    args = parser.parse_args()
    normalizer = Normalizer()
    normalizer.normalize_volumes(args.path_to_studies, args.path_to_output)
