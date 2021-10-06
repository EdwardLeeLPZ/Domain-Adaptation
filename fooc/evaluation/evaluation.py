import itertools
import numpy as np
from tabulate import tabulate

from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table


class ExtendedCOCOEvaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_50 = []
        results_per_category_75 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision_50 = precisions[0, :, idx, 0, -1]
            precision_75 = precisions[5, :, idx, 0, -1]
            precision = precision[precision > -1]
            precision_50 = precision_50[precision_50 > -1]
            precision_75 = precision_75[precision_75 > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            ap_50 = np.mean(precision_50) if precision_50.size else float("nan")
            ap_75 = np.mean(precision_75) if precision_75.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            results_per_category_50.append(("{}".format(name), float(ap_50 * 100)))
            results_per_category_75.append(("{}".format(name), float(ap_75 * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_flatten_50 = list(itertools.chain(*results_per_category_50))
        results_flatten_75 = list(itertools.chain(*results_per_category_75))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        results_2d_50 = itertools.zip_longest(*[results_flatten_50[i::N_COLS] for i in range(N_COLS)])
        results_2d_75 = itertools.zip_longest(*[results_flatten_75[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        table_50 = tabulate(
            results_2d_50,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        table_75 = tabulate(
            results_2d_75,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP75"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table_50)
        self._logger.info("Per-category {} AP75: \n".format(iou_type) + table_75)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results
