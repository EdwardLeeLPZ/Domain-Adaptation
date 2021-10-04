import copy
import torch
import numpy as np
import pycocotools.mask as mask_util

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

from fvcore.common.file_io import PathManager
from detectron2.data import DatasetMapper
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


from .datasets.register_datasets import cityscapes_sample_to_dict


class DatasetMapperWrapper(DatasetMapper):
    """
    Wraps the original detectron2 DatasetMapper to map additional domain labels.
    """
    def __init__(self, cfg, is_train=True):
        """"""
        super().__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        """"""
        dataset_dict = super().__call__(dataset_dict)
        # DA specialization
        # add domain label to instances
        if "domain" in dataset_dict:
            dataset_dict['domain'] = torch.as_tensor(
                dataset_dict["domain"], dtype=torch.float
            )
            instances = dataset_dict['instances']
            if len(instances) > 0:
                instances.gt_domains = dataset_dict['domain'].repeat(len(instances))

        return dataset_dict


class ExtendedDatasetMapper(DatasetMapper):
    """
    Extends the original detectron2 dataset mapper to support mapping of domain labels
    and 3D ground truth, if they exist.

    Args:
        See :class:`detectron2.data.DatasetMapper`.
    """
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)

        if cfg.FOOC.AUGMENTATIONS:
            self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        else:
            self.tfm_gens = []

        self.clip_boxes = cfg.DATALOADER.CLIP_BOXES_TO_IMAGE_SIZE

    def __call__(self, dataset_dict):
        """
        Pretty much copy paste from super(), except that we will be using a custom
        method rather than
        :meth:'detectron2.data.detection_utils.annotations_to_instances'
        so we can add handling for our additional ground truth information.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format, clip_boxes=self.clip_boxes
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # DA specialization
        # add domain label to instances
        if "domain" in dataset_dict:
            dataset_dict['domain'] = torch.as_tensor(
                dataset_dict["domain"], dtype=torch.float
            )
            instances = dataset_dict['instances']
            if len(instances) > 0:
                instances.gt_domains = dataset_dict['domain'].repeat(len(instances))

        return dataset_dict


class OnlineDatasetMapper(DatasetMapper):
    """
    A callable which loads images and annotations from a tuple of filenames and applies all
    mapping in an online fashion.

    The main difference to the default detectron2 dataset mapper is that it does not assume
    annotations have already been pre-processed (i.e. loaded into numpy arrays, filtered for invalid
    annotations, extraction of bounding boxes etc.), but does so on-the-fly. This is intended to
    reduce startup time of experiments but may slow down training.
    """
    def __init__(self, cfg, is_train=True):
        """"""
        super().__init__(cfg, is_train)
        self._cfg = cfg
        if self._cfg.SUDA.DATASETS.LABEL_SPACE != "Cityscapes":
            err_msg = "Online mapping has not yet been ported to work with flexible label spaces."
            raise NotImplementedError(err_msg)
        raise DeprecationWarning("This is no longer supported.")

    def __call__(self, files):
        """"""
        # FIXME: This is ugly. Mapper needs to infer the dataset from the filenames to map with
        # the respective dataset specific options. Redesign pipeline so this is not necessary.
        dataset = self._infer_dataset_from_filenames(files)
        dataset_opts = getattr(self._cfg.SUDA.DATASETS, dataset)

        # load annotations, extract bounding boxes etc.
        dataset_dict = cityscapes_sample_to_dict(files, dataset_opts.DOMAIN, dataset_opts.LOAD_MASKS)

        # Map cityscapes ids to contiguous ids
        from cityscapesscripts.helpers.labels import labels
        labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
        dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
        for anno in dataset_dict["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]

        # Apply default detectron2 mapper
        dataset_dict = super().__call__(dataset_dict)

        # map domain label to tensor
        if "domain" in dataset_dict:
            dataset_dict['domain'] = torch.as_tensor(
                dataset_dict["domain"], dtype=torch.float
            )
            instances = dataset_dict['instances']
            if instances.num_instances > 0:
                instances.gt_domains = dataset_dict['domain'].repeat(len(instances))
            else:
                instances.gt_domains = dataset_dict["domain"]

        return dataset_dict

    def _infer_dataset_from_filenames(self, files):
        img_file, _, _, _ = files

        if "cityscapes" in img_file:
            return "CITYSCAPES"
        elif "synscapes" in img_file:
            return "SYNSCAPES"
        else:
            raise ValueError("Could not infer dataset from filenames.")


def annotations_to_instances(annos, image_size, mask_format="polygon", clip_boxes=True):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    if clip_boxes:
        boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target