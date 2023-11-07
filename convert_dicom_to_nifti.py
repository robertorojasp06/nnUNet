import argparse
import dicom2nifti
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm


def _check_output_filename(output_filename):
    if not output_filename.endswith('.nii.gz'):
        raise ValueError("output filename must have the '.nii.gz' extension.")


def dcm2nii(path_to_series, path_to_output_folder, output_filename=None):
    """Convert a DICOM series to NIfTI.

        Parameters
        ----------
        path_to_series: str
            Path to the folder containing the DICOM files.
        path_to_output_folder: str
            Path to the directory to save the converted output file.
        output_filename: str
            By default, the output filename is given by the 'StudyDescription'
            attribute of the DICOM files. You can set this parameter to specify
            other filename. Make sure the extension is '.nii.gz'.
    """
    # check if series is single slice
    if len(list(Path(path_to_series).glob('*.dcm'))) == 1:
        print("Skipped. Conversion of single slice series is not supported yet.")
        return

    if output_filename:
        _check_output_filename(output_filename)
        path_to_output = Path(path_to_output_folder) / output_filename
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dicom2nifti.convert_directory(
                path_to_series,
                tmp,
                compression=True,
                reorient=True
            )
            tmp_nii = next(tmp.glob('*.nii.gz'))
            shutil.copy(tmp_nii, path_to_output)
    else:
        dicom2nifti.convert_directory(
            path_to_series,
            path_to_output_folder,
            compression=True,
            reorient=True
        )


def convert_patient_data(path_to_patient, path_to_output):
    path_to_patient = Path(path_to_patient)
    path_to_output = Path(path_to_output)
    paths_to_studies = [
        item
        for item in path_to_patient.iterdir()
        if item.is_dir()
    ]
    for path_to_study in paths_to_studies:
        print(f"study: {path_to_study.name}")
        path_to_output_study = path_to_output / path_to_study.name
        path_to_output_study.mkdir(parents=True, exist_ok=True)
        paths_to_series = [
            item
            for item in path_to_study.iterdir()
            if item.is_dir()
        ]
        for path in paths_to_series:
            dcm2nii(path, path_to_output_study)


def convert_data(path_to_data, path_to_output):
    paths_to_patients = [
        item
        for item in Path(path_to_data).iterdir()
        if item.is_dir()
    ]
    for path_to_patient in tqdm(paths_to_patients):
        convert_patient_data(
            path_to_patient,
            Path(path_to_output) / path_to_patient.name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert a set of DICOM studies from different patients
        to NIfTI format.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_data',
        type=str,
        help="""Path to the data with the following structure:
        patients -> studies -> series -> DICOM files."""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the directory to save output files."
    )
    args = parser.parse_args()
    convert_data(args.path_to_data, args.path_to_output)
