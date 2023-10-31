import argparse
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


class MSDConverter:
    """Convert datasets from Medical Segmentation Decathlon into nnunet format."""
    def __init__(self) -> None:
        self.liver_params = {
            'nnunet_dataset_id': 503,
            'task_name': 'Task03_Liver',
            'labels': {
                "background": 0,
                "tumor": 1
            },
            'description': (
                "CT studies from dataset Task03_Liver hosted "
                "by the Medical Segmentation Decathlon."
            )
        }
        self.lung_params = {
            'nnunet_dataset_id': 506,
            'task_name': 'Task06_Lung',
            'labels': {
                "background": 0,
                "tumor": 1
            },
            'description': (
                "CT studies from dataset Task06_Lung hosted "
                "by the Medical Segmentation Decathlon."
            )
        }
        self.pancreas_params = {
            'nnunet_dataset_id': 507,
            'task_name': 'Task07_Pancreas',
            'labels': {
                "background": 0,
                "tumor": 1
            },
            'description': (
                "CT studies from dataset Task07_Pancreas hosted "
                "by the Medical Segmentation Decathlon."
            )
        }
        self.colon_params = {
            'nnunet_dataset_id': 510,
            'task_name': 'Task10_Lung',
            'labels': {
                "background": 0,
                "tumor": 1
            },
            'description': (
                "CT studies from dataset Task10_Colon hosted "
                "by the Medical Segmentation Decathlon."
            )
        }
        self.fname_extension = '.nii.gz'

    def convert_dataset(self, path_to_cts, path_to_masks, seed=None, **kwargs):
        foldername = f"Dataset{kwargs['nnunet_dataset_id']:03d}_{kwargs['task_name']}"
        # Set up output folders
        path_to_output_base = Path(nnUNet_raw) / foldername
        path_to_output_cts = path_to_output_base / "imagesTr"
        path_to_output_masks = path_to_output_base / "labelsTr"
        path_to_output_cts.mkdir(parents=True, exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        # Copy data volumen with proper filename format
        studies_count = 0
        identifiers = []
        studies = Path(path_to_cts).glob(f"*{self.fname_extension}")
        for source_ct in studies:
            source_mask = Path(path_to_masks) / source_ct.name
            identifier = f"{source_ct.name.split(self.fname_extension)[0]}"
            identifiers.append(identifier)
            dest_ct = path_to_output_cts / f"{identifier}_0000{self.fname_extension}"
            dest_mask = path_to_output_masks / f"{identifier}{self.fname_extension}"       
            shutil.copy(source_ct, dest_ct)
            shutil.copy(source_mask, dest_mask)
            studies_count += 1
        # Generate JSON containing metadata required for training
        channel_names = {
            0: "CT"
        }
        generate_dataset_json(
            path_to_output_base,
            channel_names,
            labels=kwargs['labels'],
            num_training_cases=studies_count,
            file_ending='.nii.gz',
            dataset_name=kwargs['task_name'],
            reference='https://autopet-ii.grand-challenge.org/',
            description=kwargs['description']
        )
        # Manual split
        splits = []
        if not seed:
            seed = random.randint(0, 2**16 - 1)
        random.seed(seed)
        random.shuffle(identifiers)
        for fold in tqdm(range(5)):
            val_studies = identifiers[fold :: 5]
            splits.append(
                {
                    'train': [i for i in identifiers if i not in val_studies],
                    'val': val_studies
                }
            )
        path_to_output = Path(nnUNet_preprocessed) / foldername
        path_to_output.mkdir(exist_ok=True)
        with open(path_to_output / 'splits_final.json', 'w') as f:
            json.dump(splits, f, sort_keys=False, indent=4)

    def convert_liver(self, path_to_cts, path_to_masks, seed):
        self.convert_dataset(
            path_to_cts,
            path_to_masks,
            seed,
            **self.liver_params
        )

    def convert_lung(self, path_to_cts, path_to_masks, seed):
        self.convert_dataset(
            path_to_cts,
            path_to_masks,
            seed,
            **self.lung_params
        )

    def convert_pancreas(self, path_to_cts, path_to_masks, seed):
        self.convert_dataset(
            path_to_cts,
            path_to_masks,
            seed,
            **self.pancreas_params
        )

    def convert_colon(self, path_to_cts, path_to_masks, seed):
        self.convert_dataset(
            path_to_cts,
            path_to_masks,
            seed,
            **self.colon_params
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert original CT datasets into the nnUnet format.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_cts',
        type=str,
        help="""Path to the folder containing original CT volumes in
        NIfTI format."""
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="""Path to the folder containing original mask volumes in
        NIfTI format."""
    )
    parser.add_argument(
        'dataset',
        choices=['msd-liver', 'msd-lung', 'msd-pancreas', 'msd-colon'],
        help="""Dataset to be converted. This option sets the parameters
        for conversion."""
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help="""Seed value for reproducibility in manual split for cross
        validation."""
    )
    args = parser.parse_args()
    converter = MSDConverter()
    if args.dataset == 'msd-liver':
        converter.convert_liver(
            args.path_to_cts,
            args.path_to_masks,
            args.seed
        )
    elif args.dataset == 'msd-lung':
        converter.convert_lung(
            args.path_to_cts,
            args.path_to_masks,
            args.seed
        )
    elif args.dataset == 'msd-pancreas':
        converter.convert_pancreas(
            args.path_to_cts,
            args.path_to_masks,
            args.seed
        )
    elif args.dataset == 'msd-colon':
        converter.convert_colon(
            args.path_to_cts,
            args.path_to_masks,
            args.seed
        )
