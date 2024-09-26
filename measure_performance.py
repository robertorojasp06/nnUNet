import numpy as np
import SimpleITK as sitk
import json
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Evaluator:
    def __init__(self) -> None:
        self.results = None
        self.metadata = None
        self.verbose = False

    def _compute_base_metrics(self, gt_array, prediction_array, label_value):
        gt_array_bool = gt_array == label_value
        prediction_array_bool = prediction_array == label_value
        intersection_array = np.logical_and(gt_array_bool, prediction_array_bool)
        TP = np.sum(intersection_array)
        FP = np.sum(prediction_array_bool) - np.sum(np.logical_and(prediction_array_bool, intersection_array))
        FN = np.sum(gt_array_bool) - np.sum(np.logical_and(gt_array_bool, intersection_array))
        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "n_ref": np.sum(gt_array_bool),
            "n_predicted": np.sum(prediction_array_bool)
        }

    def _process_sample(self, sample):
        if self.verbose:
            print(f"processing {sample['path_to_gt'].name} ...")
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(sample["path_to_gt"]))
        prediction_array = sitk.GetArrayFromImage(sitk.ReadImage(sample["path_to_prediction"]))
        output_results = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in sample.items()
        }
        output_results.update({"performance": []})
        foreground_labels = {
            key: value
            for key, value in self.metadata["labels"].items()
            if value > 0
        }
        for label_name, label_value in foreground_labels.items():
            base_metrics = self._compute_base_metrics(
                gt_array,
                prediction_array,
                label_value
            )
            output_results["performance"].append({
                "label_name": label_name,
                "label_value": label_value,
                **base_metrics,
                "dice": float(2 * base_metrics["TP"] / (2 * base_metrics["TP"] + base_metrics["FP"] + base_metrics["FN"])) if base_metrics["n_ref"] > 0 else None
            })
        self.results.append(output_results)

    def measure_performance(self, path_to_gts, path_to_predictions,
                            path_to_json, max_workers=4):
        with open(path_to_json, 'r') as file:
            metadata = json.load(file)
        samples = [
            {
                "path_to_gt": Path(path_to_gts) / path_to_gt.name,
                "path_to_prediction": Path(path_to_predictions) / path_to_gt.name
            }
            for path_to_gt in Path(path_to_gts).glob(f"*{metadata['file_ending']}")
        ]
        self.results = []
        self.metadata = metadata
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._process_sample, samples), total=len(samples)))
        return self.results

def convert_to_df(results):
    results_df = [
        {
            "path_to_gt": item["path_to_gt"],
            "path_to_prediction": item["path_to_prediction"],
            **perf
        }
        for item in results
        for perf in item["performance"]
    ]
    return pd.DataFrame(results_df)


def main():
    parser = argparse.ArgumentParser(
        description="Measure segmentation performance of predicted CTs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_gts',
        type=str,
        help="""Path to the directory containing the CT files with
        ground truth segmentations."""
    )
    parser.add_argument(
        'path_to_predictions',
        type=str,
        help="""Path to the directory containing the CT files with
        predicted segmentations."""
    )
    parser.add_argument(
        'path_to_json',
        type=str,
        help="""Path to the JSON file containing the dictionary 'labels'
        and 'file_ending'. Follows the structure of the file 'dataset.json'"""
    )
    parser.add_argument(
        'path_to_output',
        type=str,
        help="Path to the output directory."
    )
    args = parser.parse_args()
    evaluator = Evaluator()
    results = evaluator.measure_performance(
        args.path_to_gts,
        args.path_to_predictions,
        args.path_to_json,
        max_workers=8
    )
    with open(Path(args.path_to_output) / "performance.json", 'w') as file:
        json.dump(results, file, indent=4, cls=NumpyEncoder)
    results_df = convert_to_df(results)
    results_df.to_csv(
        Path(args.path_to_output) / "performance.csv",
        index=False
    )


if __name__ == "__main__":
    main()
