import argparse
import json
import shutil
import random
import SimpleITK as sitk
import numpy as np
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
            ),
            'original_tumor_label': 2
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
            ),
            'original_tumor_label': 1
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
            ),
            'original_tumor_label': 2
        }
        self.colon_params = {
            'nnunet_dataset_id': 510,
            'task_name': 'Task10_Colon',
            'labels': {
                "background": 0,
                "tumor": 1
            },
            'description': (
                "CT studies from dataset Task10_Colon hosted "
                "by the Medical Segmentation Decathlon."
            ),
            'original_tumor_label': 1
        }
        self.fname_extension = '.nii.gz'

    def _convert_mask(self, path_to_mask, original_label, final_label):
        """Return mask as a nifti image only with tumor label"""
        mask_img = sitk.ReadImage(path_to_mask)
        mask_array = sitk.GetArrayFromImage(mask_img)
        new_mask_array = ((mask_array == original_label) * final_label).astype(mask_array.dtype)
        new_img = sitk.GetImageFromArray(new_mask_array)
        new_img.CopyInformation(mask_img)
        return new_img

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
        studies = list(Path(path_to_cts).glob(f"*{self.fname_extension}"))
        for source_ct in tqdm(studies):
            source_mask = Path(path_to_masks) / source_ct.name
            identifier = f"{source_ct.name.split(self.fname_extension)[0]}"
            identifiers.append(identifier)
            dest_ct = path_to_output_cts / f"{identifier}_0000{self.fname_extension}"
            dest_mask = path_to_output_masks / f"{identifier}{self.fname_extension}"
            converted_mask_img = self._convert_mask(
                source_mask,
                kwargs['original_tumor_label'],
                kwargs['labels']['tumor']
            )
            shutil.copy(source_ct, dest_ct)
            sitk.WriteImage(converted_mask_img, dest_mask)
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


class KiTS19Converter:
    """Convert dataset from Kidney Tumor Segmentation (KiTS19) into nnunet format."""
    def __init__(self) -> None:
        self.params = {
            'nnunet_dataset_id': 511,
            'task_name': 'KiTS19_Kidney',
            'labels': {
                "background": 0,
                "kidney": 1,
                "tumor": 2
            },
            'description': (
                "CT studies from dataset kits19 hosted "
                "in the github repository 'https://github.com/neheller/kits19'."
            )
        }
        self.fname_extension = '.nii.gz'

    def convert_dataset(self, path_to_studies, seed=None):
        foldername = f"Dataset{self.params['nnunet_dataset_id']:03d}_{self.params['task_name']}"
        # Set up output folders
        path_to_output_base = Path(nnUNet_raw) / foldername
        path_to_output_cts = path_to_output_base / "imagesTr"
        path_to_output_masks = path_to_output_base / "labelsTr"
        path_to_output_cts.mkdir(parents=True, exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        # Copy data volumen with proper filename format
        studies_count = 0
        identifiers = []
        studies = [
            item
            for item in Path(path_to_studies).iterdir()
            if item.is_dir() and len(list(item.glob(f"*{self.fname_extension}"))) == 2
        ]
        for study in tqdm(studies):
            source_ct = study / 'imaging.nii.gz'
            source_mask = study / 'segmentation.nii.gz'
            identifier = study.name
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
            labels=self.params['labels'],
            num_training_cases=studies_count,
            file_ending=self.fname_extension,
            dataset_name=self.params['task_name'],
            reference='https://github.com/neheller/kits19',
            description=self.params['description']
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


class CTLymphNodesConverter:
    """Convert dataset CT Lymph Nodes into nnunet format."""
    def __init__(self) -> None:
        self.params = {
            'nnunet_dataset_id': 512,
            'task_name': 'CTLN_LymphNodes',
            'labels': {
                "background": 0,
                "lymph_node": 1
            },
            'description': (
                "CT studies from dataset CT Lymph Nodes hosted "
                "in the TCIA repository https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546"
            )
        }
        self.fname_extension = '.nii.gz'

    def convert_dataset(self, path_to_cts, path_to_masks, seed=None):
        foldername = f"Dataset{self.params['nnunet_dataset_id']:03d}_{self.params['task_name']}"
        # Set up output folders
        path_to_output_base = Path(nnUNet_raw) / foldername
        path_to_output_cts = path_to_output_base / "imagesTr"
        path_to_output_masks = path_to_output_base / "labelsTr"
        path_to_output_cts.mkdir(parents=True, exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        # Copy data volumen with proper filename format
        studies_count = 0
        identifiers = []
        studies = list(Path(path_to_cts).glob(f"*{self.fname_extension}"))
        for source_ct in tqdm(studies):
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
            labels=self.params['labels'],
            num_training_cases=studies_count,
            file_ending='.nii.gz',
            dataset_name=self.params['task_name'],
            reference='https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546',
            description=self.params['description']
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert original CT datasets into the nnUnet format.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_cts',
        type=str,
        help="""Path to the folder containing original CT volumes in
        NIfTI format. Path the to the directory containing the studies folders
        if that is the format of the original dataset."""
    )
    parser.add_argument(
        'dataset',
        choices=[
            'msd-liver', 'msd-lung', 'msd-pancreas', 'msd-colon',
            'kits19', 'ct-lymph-nodes'
        ],
        help="""Dataset to be converted. This option sets the parameters
        for conversion."""
    )
    parser.add_argument(
        '--path_to_masks',
        type=str,
        default=None,
        help="""Path to the folder containing original mask volumes in
        NIfTI format."""
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help="""Seed value for reproducibility in manual split for cross
        validation."""
    )
    args = parser.parse_args()
    if args.dataset in ('msd-liver', 'msd-lung', 'msd-pancreas', 'msd-colon'):
        if not args.path_to_masks:
            raise ValueError("Path to mask volumes must be specified.")
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
    elif args.dataset == 'kits19':
        converter = KiTS19Converter()
        converter.convert_dataset(args.path_to_cts, args.seed)
    elif args.dataset == 'ct-lymph-nodes':
        if not args.path_to_masks:
            raise ValueError("Path to mask volumes must be specified.")
        converter = CTLymphNodesConverter()
        converter.convert_dataset(
            args.path_to_cts,
            args.path_to_masks,
            args.seed
        )
