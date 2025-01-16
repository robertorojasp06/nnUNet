import numpy as np
import SimpleITK as sitk
import json
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union, Dict

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


WINDOWS = {
	"lung": {"L": -500, "W": 1400},
	"abdomen": {"L": 40, "W": 350},
	"bone": {"L": 400, "W": 1000},
	"air": {"L": -426, "W": 1000},
	"brain": {"L": 50, "W": 100},
	"mediastinum": {"L": 50, "W": 350}
}


def get_windows_mapping(window_arg: str, path_to_cts: str):
	if window_arg not in WINDOWS:
		with open(window_arg, 'r') as file:
			mapping = json.load(file)
	else:
		mapping = {
			path.name: window_arg
			for path in Path(path_to_cts).glob('*.nii.gz')
		}
	return mapping


def check_windows_mapping(mapping: Dict[str, str], path_to_cts: str):
	# Check wrong windows
	wrong_windows = [
		f"filename '{filename}' with wrong window '{window}'."
		for filename, window in mapping.items()
		if window not in WINDOWS
	]
	if wrong_windows:
		raise ValueError('\n'.join(wrong_windows))
	# Check all CTs have their corresponding window
	unassigned_cts = [
		f"filename '{path.name}' does not have a window assigned."
		for path in Path(path_to_cts).glob('*.nii.gz')
		if path.name not in mapping.keys()
	]
	if unassigned_cts:
		raise ValueError('\n'.join(unassigned_cts))


def normalize_ct(
    ct_array: np.ndarray,
    window: Optional[Dict[str, Union[int, float]]] = None,
    epsilon: float = 1e-6
) -> np.ndarray:
    if window:
        lower_bound = window["L"] - window["W"] / 2
        upper_bound = window["L"] + window["W"] / 2
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
    else:
        lower_bound= np.percentile(ct_array[ct_array > 0], 0.5)
        upper_bound = np.percentile(ct_array[ct_array > 0], 99.5)
        ct_array_pre = np.clip(ct_array, lower_bound, upper_bound)
        ct_array_pre = (
            (ct_array_pre - np.min(ct_array_pre) + epsilon)
            / (np.max(ct_array_pre) - np.min(ct_array_pre) + epsilon)
            * 255.0
        )
        ct_array_pre[ct_array == 0] = 0
    return np.uint8(ct_array_pre)


class HcuchConverter:
    """Convert format of HCUCH dataset into nnunet format."""
    def __init__(self, dataset_id=513, task_name='HCUCH_Lesions',
                 channel_name="CT", unique_label=None,
                 extra_description=None) -> None:
        self.params = {
            'nnunet_dataset_id': dataset_id,
            'task_name': task_name,
            'channel_names': {
                0: channel_name
            },
            'description': (
                f"CT studies from HCUCH - FONDEF ID23|10337. {extra_description}"
                if extra_description
                else "CT studies from HCUCH - FONDEF ID23|10337"
            )
        }
        self.fname_extension = '.nii.gz'
        self.unique_label = unique_label

    @property
    def foldername(self):
        return f"Dataset{self.params['nnunet_dataset_id']:03d}_{self.params['task_name']}"

    @property
    def path_to_output_base(self):
        return Path(nnUNet_raw) / self.foldername

    @property
    def unique_label(self):
         return self._unique_label

    @unique_label.setter
    def unique_label(self, value):
        assert value in ["tumor", "adenopathy", None]
        if value == 'tumor':
             self.params['labels'] = {
                "background": 0,
                "tumor": 1
            }
        elif value == 'adenopathy':
             self.params['labels'] = {
                "background": 0,
                "adenopathy": 1
            }
        else:
             self.params['labels'] = {
                "background": 0,
                "tumor": 1,
                "adenopathy": 2
            }
        self._unique_label = value

    def _transform_mask(self, path_to_mask, path_to_mask_labels,
                        requested_labels=None):
        # Find all integers belonging to tumor and adenopathy
        label_groups = {
            "tumor": [],
            "adenopathy": []
        }
        with open(path_to_mask_labels, 'r') as file:
            labels = json.load(file)
        # Filter labels according to requested labels
        labels = {
            key: value
            for key, value in labels.items()
            if ','.join([item.strip() for item in value.split(',')]) in requested_labels
        }
        if not labels:
            mask_image = sitk.ReadImage(path_to_mask)
            mask_array = sitk.GetArrayFromImage(mask_image)
            final_mask_array = np.zeros(mask_array.shape, dtype=mask_array.dtype)
            final_mask_image = sitk.GetImageFromArray(final_mask_array)
            final_mask_image.CopyInformation(mask_image)
            return final_mask_image
        for label_value, label_name in labels.items():
            if label_name.split(',')[0] in ['p', 'm']:
                label_groups['tumor'].append(int(label_value))
            elif label_name.split(',')[0] == 'n':
                label_groups['adenopathy'].append(int(label_value))
            else:
                raise ValueError(f"{path_to_mask_labels.name} has unexpected label name: {label_name}")
        # Transform mask to have "tumor" and "adenopathy" label values
        mask_image = sitk.ReadImage(path_to_mask)
        mask_array = sitk.GetArrayFromImage(mask_image)
        final_mask_array = np.zeros(mask_array.shape, dtype=mask_array.dtype)
        if self.unique_label:
             label_groups = {
                  key: value
                  for key, value in label_groups.items()
                  if key == self.unique_label
             }
        for label_name, label_values in label_groups.items():
            indices = np.isin(mask_array, label_values)
            final_mask_array[indices] = self.params['labels'][label_name]
        final_mask_image = sitk.GetImageFromArray(final_mask_array)
        final_mask_image.CopyInformation(mask_image)
        return final_mask_image

    def convert_dataset(self, path_to_cts, path_to_masks, path_to_labels,
                        windows_mapping=None, seed=None,
                        requested_labels=None):
        # Set up output folders
        path_to_output_cts = self.path_to_output_base / "imagesTr"
        path_to_output_masks = self.path_to_output_base / "labelsTr"
        path_to_output_cts.mkdir(parents=True, exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        identifiers = []
        for path_to_source_ct in tqdm(list(Path(path_to_cts).glob(f"*{self.fname_extension}"))):
            # Update mask according to the labels
            converted_mask_image = self._transform_mask(
                Path(path_to_masks) / path_to_source_ct.name,
                Path(path_to_labels) / f"{path_to_source_ct.name.split(self.fname_extension)[0]}.json",
                requested_labels
            )
            # Full zero masks are discarded
            if sitk.GetArrayFromImage(converted_mask_image).sum() == 0:
                 continue
            # Build the CT identifier and write the converted mask
            identifier = path_to_source_ct.name.split(self.fname_extension)[0]
            identifiers.append(identifier)
            sitk.WriteImage(
                converted_mask_image,
                path_to_output_masks / f"{identifier}{self.fname_extension}"
            )
            # Copy data volumes with proper filename format (or normalize CT if required)
            path_to_dest_ct = path_to_output_cts / f"{identifier}_0000{self.fname_extension}"
            if windows_mapping:
                ct_image = sitk.ReadImage(path_to_source_ct)
                ct_array = sitk.GetArrayFromImage(ct_image)
                window_name = windows_mapping.get(path_to_source_ct.name)
                ct_array_norm = normalize_ct(ct_array, WINDOWS.get(window_name))
                ct_image_norm = sitk.GetImageFromArray(ct_array_norm)
                ct_image_norm.CopyInformation(ct_image)
                sitk.WriteImage(
                    ct_image_norm,
                    path_to_dest_ct
                )
            else:
                shutil.copy(path_to_source_ct, path_to_dest_ct)
        # Generate JSON containing metadata required for training
        generate_dataset_json(
            self.path_to_output_base,
            self.params['channel_names'],
            labels=self.params['labels'],
            num_training_cases=len(identifiers),
            file_ending=self.fname_extension,
            dataset_name=self.params['task_name'],
            reference='HCUCH data',
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
        path_to_output = Path(nnUNet_preprocessed) / self.foldername
        path_to_output.mkdir(exist_ok=True)
        with open(path_to_output / 'splits_final.json', 'w') as f:
            json.dump(splits, f, sort_keys=False, indent=4)

    def convert_test_set(self, path_to_cts, path_to_masks, path_to_labels,
                         windows_mapping=None, requested_labels=None):
        # Set up output folders
        path_to_output_cts = self.path_to_output_base / "imagesTs"
        path_to_output_masks = self.path_to_output_base / "labelsTs"
        path_to_output_cts.mkdir(parents=True, exist_ok=True)
        path_to_output_masks.mkdir(exist_ok=True)
        identifiers = []
        for path_to_source_ct in tqdm(list(Path(path_to_cts).glob(f"*{self.fname_extension}"))):
            # Update mask according to the labels
            converted_mask_image = self._transform_mask(
                Path(path_to_masks) / path_to_source_ct.name,
                Path(path_to_labels) / f"{path_to_source_ct.name.split(self.fname_extension)[0]}.json",
                requested_labels
            )
            # Full zero masks are discarded
            if sitk.GetArrayFromImage(converted_mask_image).sum() == 0:
                 continue
            # Build the CT identifier and write the converted mask
            identifier = path_to_source_ct.name.split(self.fname_extension)[0]
            identifiers.append(identifier)
            sitk.WriteImage(
                converted_mask_image,
                path_to_output_masks / f"{identifier}{self.fname_extension}"
            )
            # Copy data volumes with proper filename format (or normalize CT if required)
            path_to_dest_ct = path_to_output_cts / f"{identifier}_0000{self.fname_extension}"
            if windows_mapping:
                ct_image = sitk.ReadImage(path_to_source_ct)
                ct_array = sitk.GetArrayFromImage(ct_image)
                window_name = windows_mapping.get(path_to_source_ct.name)
                ct_array_norm = normalize_ct(ct_array, WINDOWS.get(window_name))
                ct_image_norm = sitk.GetImageFromArray(ct_array_norm)
                ct_image_norm.CopyInformation(ct_image)
                sitk.WriteImage(
                    ct_image_norm,
                    path_to_dest_ct
                )
            else:
                shutil.copy(path_to_source_ct, path_to_dest_ct)


def main():
    parser = argparse.ArgumentParser(
        description="""Convert original CT dataset from HCUCH into the nnUnet format.""",
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
        'path_to_labels',
        type=str,
        help="""Path to the folder containing JSON files with labels for
        each mask."""
    )
    parser.add_argument(
        '--path_to_test_cts',
        default=None,
        type=str,
        help="""Path to the folder containing original CT volumes in
        NIfTI format from the testing set."""
    )
    parser.add_argument(
        '--path_to_test_masks',
        default=None,
        type=str,
        help="""Path to the folder containing original mask volumes in
        NIfTI format from the testing set."""
    )
    parser.add_argument(
        '--path_to_test_labels',
        default=None,
        type=str,
        help="""Path to the folder containing JSON files with labels for
        each mask from the testing set."""
    )
    parser.add_argument(
		'--window',
		type=str,
        default=None,
		help=f"""Window for CT normalization: {list(WINDOWS.keys())}.
		This window is applied on all CTs. Alternatively, you can provide
		the path to a JSON file with a dictionary containing the
		mapping between filenames and windows."""
	)
    parser.add_argument(
         '--channel_name',
         type=str,
         default='CT',
         help="""This is a field of the dataset.json file. The specific
         name determines the normalization scheme followed by the nnUNet
         pipeline. List of currently available names: 'CT', 'noNorm',
         'rescale_to_0_1', 'rgb_to_0_1', 'zscore'. Any name not included
         produces the default 'zscore' normalization (once nnUNet is run).
         If you normalize your CTs using the 'window' parameter from this script,
         then set this parameter to 'noNorm'."""
    )
    parser.add_argument(
         '--unique_label',
         type=str,
         choices=['tumor', 'adenopathy'],
         default=None,
         help="""Set this flag to obtain masks only with annotations from
         the specified label. By default, both labels are included."""
    )
    parser.add_argument(
        '--requested_labels',
        type=str,
        nargs='+',
        default=None,
        help="""Only annotations of the specified labels are considered from
        each JSON file. If a JSON file does not have any of the specified labels,
        then the CT image and mask are discarded."""
    )
    parser.add_argument(
         '--dataset_id',
         type=int,
         default=513,
         help="Dataset ID. Used to give a name to the data folders."
    )
    parser.add_argument(
         '--task_name',
         type=str,
         default='HCUCH_Lesions',
         help="Task name. Used to give a name to the data folders."
    )
    parser.add_argument(
         '--extra_description',
         type=str,
         default=None,
         help="""Add an extra description to the main dataset
         description."""
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help="""Seed value for reproducibility in manual split for cross
        validation."""
    )
    args = parser.parse_args()
    testing_set_args = [
        args.path_to_test_cts,
        args.path_to_test_masks,
        args.path_to_test_labels
    ]
    if any(testing_set_args) and not all(testing_set_args):
        parser.error(
            "If any of --path_to_test_cts, --path_to_test_masks, "
            "or --path_to_test_labels are set, then all must be set."
        )
    if args.window:
        windows_mapping = get_windows_mapping(
            args.window,
            args.path_to_cts
        )
        check_windows_mapping(
             windows_mapping,
             args.path_to_cts
        )
    else:
         windows_mapping = None
    converter = HcuchConverter(
         args.dataset_id,
         args.task_name,
         args.channel_name,
         args.unique_label,
         args.extra_description
    )
    converter.convert_dataset(
        args.path_to_cts,
        args.path_to_masks,
        args.path_to_labels,
        windows_mapping=windows_mapping,
        seed=args.seed,
        requested_labels=args.requested_labels
    )
    if all(testing_set_args):
        converter.convert_test_set(
            args.path_to_test_cts,
            args.path_to_test_masks,
            args.path_to_test_labels,
            windows_mapping=windows_mapping,
            requested_labels=args.requested_labels
        )


if __name__ == "__main__":
    main()
