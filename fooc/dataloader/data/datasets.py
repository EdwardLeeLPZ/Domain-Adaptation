import os
import random
import functools
import logging
import json
import glob
import numpy as np
import multiprocessing as mp
import xml.etree.ElementTree as ET
from detectron2.data import MetadataCatalog, DatasetCatalog, detection_utils
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode

from .label_spaces import *

# ==================== Cityscapes ==================== #
def register_cityscapes(cfg, root_dir="datasets"):
    """
    Register Cityscapes dataset in the standard Detectron2 annotation format for
    2D instance detection.

    Args:
        cfg(config): global configs of Detectron2
        root_dir (str or path-like): directory which contains all the data.
    """
    # set the data directory and basic configs of the Cityscapes dataset
    data_dir = os.path.join(root_dir, "Cityscapes")
    domain = cfg.FOOC.DATASETS.CITYSCAPES.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.CITYSCAPES.LABEL_SPACE)

    # load valid class names of labels
    thing_classes = [label.name for label in label_space.labels
                     if label.hasInstances and not label.ignoreInEval]
    stuff_classes = [label.name for label in label_space.labels
                     if not label.hasInstances]

    # split the data set into the training set, the validation set and the test set
    # according to the preset split method, and then register them
    for split in ["train", "val", "test"]:
        DatasetCatalog.register(
            "Cityscapes_"+split,
            lambda data_dir=data_dir, split=split, config=cfg:
                load_cityscapes_samples(data_dir, split, config)
        )

        MetadataCatalog.get("Cityscapes_"+split).set(
            data_dir=data_dir,
            evaluator_type="coco",
            domain=domain,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
        )


def _get_cityscapes_files(image_dir, gt_dir, logger):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(
                city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(
                city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(
                city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_cityscapes_samples(data_dir, split, cfg):
    """
    Select appropriate samples according to the preset split method to create
    a dict of the entire dataset

    Args:
        data_dir(str): directory for storing images and labels
        split(str): dataset type ("train", "val" or "test")
        cfg(config): global configs of Detectron2

    Returns:
        A dict list of all image within a dataset in Detectron2 Dataset format.
    """
    # verify the validity of the directories
    assert os.path.exists(os.path.join(data_dir, "leftImg8bit"))
    assert os.path.exists(os.path.join(data_dir, "gtFine"))

    logger = logging.getLogger("detectron2.data.datasets.Cityscapes")
    logger.info(
        "Preprocessing Cityscapes object detection annotations of {} set...".format(split))

    # obtain file names of images and their annotation
    image_dir = os.path.join(data_dir, "leftImg8bit", split)
    gt_dir = os.path.join(data_dir, "gtFine", split)
    files = _get_cityscapes_files(image_dir, gt_dir, logger)

    # sample a certain number of image files
    if cfg.FOOC.DATASETS.CITYSCAPES.SUBSAMPLE == 0:
        err_msg = "You are trying to draw a subsample of length zero."
        raise ValueError(err_msg)
    elif cfg.FOOC.DATASETS.CITYSCAPES.SUBSAMPLE != -1:
        files = random.sample(files, cfg.FOOC.DATASETS.CITYSCAPES.SUBSAMPLE)
    assert len(files)

    # parse the gathered files into detectron2 data dict
    domain = cfg.FOOC.DATASETS.CITYSCAPES.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.CITYSCAPES.LABEL_SPACE)

    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    records = pool.map(
        functools.partial(
            cityscapes_sample_to_dict,
            is_train=True,
            domain=domain,
            label_space=label_space),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(records), data_dir))
    pool.close()

    return records


def cityscapes_sample_to_dict(file, is_train, domain, label_space):
    """
    Parse Cityscapes annotation files to a instance detection dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        is_train(bool): whether to prepare the dataset for training
        domain(str): domain of the image 
        label_space(cs_labels): Cityscapes label space

    Returns:
        A dict of 1 image in Detectron2 Dataset format.
    """
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    image_file, _, _, json_file = file

    with PathManager.open(json_file, "r") as f:
        jsonobj = json.load(f)

    # record the basic information of the image
    record = {
        "file_name": image_file,
        "image_id": os.path.basename(image_file),
        "height": jsonobj["imgHeight"],
        "width": jsonobj["imgWidth"],
    }

    # record the domain information of the image
    if is_train:
        if domain == "source":
            record["domain"] = np.zeros([1], dtype=np.float32)
        elif domain == "target":
            record["domain"] = np.ones([1], dtype=np.float32)
        elif domain is not None:
            raise ValueError(
                "The dataset domain can only be either \"source\", \"target\" or \"None.\""
            )

    annos = []
    # CityscapesScripts draw the polygons in sequential order
    # and each polygon *overwrites* existing ones. See
    # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
    # We use reverse order, and each polygon *avoids* early ones.
    # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
    for obj in jsonobj["objects"][::-1]:
        label_name = obj["label"]

        if label_space.__name__ != "Cityscapes":
            err_msg = "You are trying to map classes of {} label space into Cityscapes format.".format(
                str(label_space.__name__))
            raise ValueError(err_msg)
        else:
            try:
                label = label_space.name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = label_space.name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

        # remove instances of irrelevant categories
        if not label.hasInstances or label.ignoreInEval:
            continue

        # make the ids of the remaining categories continuous
        contiguous_id_dict = {l.id: idx for idx,
                              l in enumerate(label_space.instance_labels)}

        # Cityscapes's raw annotations uses integer coordinates
        # Therefore +0.5 here
        poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
        # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
        # polygons for evaluation. This function operates in integer space
        # and draws each pixel whose center falls into the polygon.
        # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
        # We therefore dilate the input polygon by 0.5 as our input.
        poly = Polygon(poly_coord).buffer(0.5, resolution=4)
        # get the bounding box of the polygon
        (xmin, ymin, xmax, ymax) = poly.bounds

        # only 2D information will be loaded
        anno = {
            "iscrowd": False,
            "bbox": (xmin, ymin, xmax, ymax),
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": contiguous_id_dict[label.id],
        }

        annos.append(anno)

    record["annotations"] = annos
    return record

# ==================== Foggy Cityscapes ==================== #
def register_foggy_cityscapes(cfg, root_dir="datasets"):
    """
    Register Foggy Cityscapes dataset in the standard Detectron2 annotation format for
    2D instance detection.

    Args:
        cfg(config): global configs of Detectron2
        root_dir (str or path-like): directory which contains all the data.
    """
    # set the data directory and basic configs of the Cityscapes dataset
    data_dir = os.path.join(root_dir, "FoggyCityscapes")
    domain = cfg.FOOC.DATASETS.FOGGYCITYSCAPES.DOMAIN
    label_space = get_label_space(
        cfg.FOOC.DATASETS.FOGGYCITYSCAPES.LABEL_SPACE)

    # load valid class names of labels
    thing_classes = [label.name for label in label_space.labels
                     if label.hasInstances and not label.ignoreInEval]
    stuff_classes = [label.name for label in label_space.labels
                     if not label.hasInstances]

    # split the data set into the training set, the validation set and the test set
    # according to the preset split method, and then register them
    for split in ["train", "val", "test"]:
        DatasetCatalog.register(
            "FoggyCityscapes_"+split,
            lambda data_dir=data_dir, split=split, config=cfg:
                load_foggy_cityscapes_samples(data_dir, split, config)
        )

        MetadataCatalog.get("FoggyCityscapes_"+split).set(
            data_dir=data_dir,
            evaluator_type="coco",
            domain=domain,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
        )


def _get_foggy_cityscapes_files(image_dir, gt_dir, beta, logger):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for image_file in glob.glob(os.path.join(city_img_dir, f"*_foggy_beta_{beta}.png")):

            suffix = f"leftImg8bit_foggy_beta_{beta}.png"
            assert image_file.endswith(suffix), image_file
            basename = image_file[len(city_img_dir)+1: -len(suffix)]

            instance_file = os.path.join(
                city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(
                city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(
                city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_foggy_cityscapes_samples(data_dir, split, cfg):
    """
    Select appropriate samples according to the preset split method to create
    a dict of the entire dataset

    Args:
        data_dir(str): directory for storing images and labels
        split(str): dataset type ("train", "val" or "test")
        cfg(config): global configs of Detectron2

    Returns:
        A dict list of all image within a dataset in Detectron2 Dataset format.
    """
    # verify the validity of the directories
    assert os.path.exists(os.path.join(data_dir, "leftImg8bit_foggy"))
    assert os.path.exists(os.path.join(data_dir, "gtFine"))

    logger = logging.getLogger("detectron2.data.datasets.FoggyCityscapes")
    logger.info(
        "Preprocessing FoggyCityscapes object detection annotations of {} set...".format(split))

    # obtain file names of images and their annotation
    image_dir = os.path.join(data_dir, "leftImg8bit_foggy", split)
    gt_dir = os.path.join(data_dir, "gtFine", split)
    beta = cfg.FOOC.DATASETS.FOGGYCITYSCAPES.BETA
    files = _get_foggy_cityscapes_files(image_dir, gt_dir, beta, logger)

    # sample a certain number of image files
    if cfg.FOOC.DATASETS.FOGGYCITYSCAPES.SUBSAMPLE == 0:
        err_msg = "You are trying to draw a subsample of length zero."
        raise ValueError(err_msg)
    elif cfg.FOOC.DATASETS.FOGGYCITYSCAPES.SUBSAMPLE != -1:
        files = random.sample(
            files, cfg.FOOC.DATASETS.FOGGYCITYSCAPES.SUBSAMPLE)
    assert len(files)

    # parse the gathered files into detectron2 data dict
    domain = cfg.FOOC.DATASETS.FOGGYCITYSCAPES.DOMAIN
    label_space = get_label_space(
        cfg.FOOC.DATASETS.FOGGYCITYSCAPES.LABEL_SPACE)

    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    records = pool.map(
        functools.partial(
            foggy_cityscapes_sample_to_dict,
            is_train=True,
            domain=domain,
            label_space=label_space),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(records), data_dir))
    pool.close()

    return records


def foggy_cityscapes_sample_to_dict(file, is_train, domain, label_space):
    """
    Parse Foggy Cityscapes annotation files to a instance detection dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        is_train(bool): whether to prepare the dataset for training
        domain(str): domain of the image 
        label_space(cs_labels): Foggy Cityscapes label space

    Returns:
        A dict of 1 image in Detectron2 Dataset format.
    """
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    image_file, _, _, json_file = file

    with PathManager.open(json_file, "r") as f:
        jsonobj = json.load(f)

    # record the basic information of the image
    record = {
        "file_name": image_file,
        "image_id": os.path.basename(image_file),
        "height": jsonobj["imgHeight"],
        "width": jsonobj["imgWidth"],
    }

    # record the domain information of the image
    if is_train:
        if domain == "source":
            record["domain"] = np.zeros([1], dtype=np.float32)
        elif domain == "target":
            record["domain"] = np.ones([1], dtype=np.float32)
        elif domain is not None:
            raise ValueError(
                "The dataset domain can only be either \"source\", \"target\" or \"None.\""
            )

    annos = []
    # CityscapesScripts draw the polygons in sequential order
    # and each polygon *overwrites* existing ones. See
    # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
    # We use reverse order, and each polygon *avoids* early ones.
    # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
    for obj in jsonobj["objects"][::-1]:
        label_name = obj["label"]

        if label_space.__name__ != "Cityscapes":
            err_msg = "You are trying to map classes of {} label space into Cityscapes format.".format(
                str(label_space.__name__))
            raise ValueError(err_msg)
        else:
            try:
                label = label_space.name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = label_space.name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

        # remove instances of irrelevant categories
        if not label.hasInstances or label.ignoreInEval:
            continue

        # make the ids of the remaining categories continuous
        contiguous_id_dict = {l.id: idx for idx,
                              l in enumerate(label_space.instance_labels)}

        # Cityscapes's raw annotations uses integer coordinates
        # Therefore +0.5 here
        poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
        # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
        # polygons for evaluation. This function operates in integer space
        # and draws each pixel whose center falls into the polygon.
        # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
        # We therefore dilate the input polygon by 0.5 as our input.
        poly = Polygon(poly_coord).buffer(0.5, resolution=4)
        # get the bounding box of the polygon
        (xmin, ymin, xmax, ymax) = poly.bounds

        # only 2D information will be loaded
        anno = {
            "iscrowd": False,
            "bbox": (xmin, ymin, xmax, ymax),
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": contiguous_id_dict[label.id],
        }

        annos.append(anno)

    record["annotations"] = annos
    return record

# ==================== Sim10k ==================== #
def register_sim10k(cfg, root_dir="datasets"):
    """
    Register SIM10k dataset in the standard Detectron2 annotation format for
    2D instance detection.

    Args:
        cfg(config): global configs of Detectron2
        root_dir (str or path-like): directory which contains all the data.
    """
    # set the data directory and basic configs of the SIM10k dataset
    data_dir = os.path.join(root_dir, "Sim10k")
    domain = cfg.FOOC.DATASETS.SIM10K.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.SIM10K.LABEL_SPACE)

    # load valid class names of labels
    thing_classes = [label.name for label in label_space.labels
                     if label.hasInstances and not label.ignoreInEval]
    stuff_classes = [label.name for label in label_space.labels
                     if not label.hasInstances]

    # split the data set into the training set and the validation set according to the preset split method, and then register them
    for split in ["train", "val"]:
        DatasetCatalog.register(
            "Sim10k_"+split,
            lambda data_dir=data_dir, split=split, config=cfg:
                load_sim10k_samples(data_dir, split, config)
        )

        MetadataCatalog.get("Sim10k_"+split).set(
            data_dir=data_dir,
            evaluator_type="coco",
            domain=domain,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
        )


def load_sim10k_samples(data_dir, split, cfg):
    """
    Select appropriate samples according to the preset split method to create
    a dict of the entire dataset

    Args:
        data_dir(str): directory for storing images and labels
        split(str): dataset type ("train" or "val")
        cfg(config): global configs of Detectron2

    Returns:
        A dict list of all image within a dataset in Detectron2 Dataset format.
    """
    # verify the validity of the directories
    assert os.path.exists(os.path.join(data_dir, "VOC2012", "JPEGImages"))
    assert os.path.exists(os.path.join(data_dir, "VOC2012", "Annotations"))
    assert os.path.exists(os.path.join(data_dir, "VOC2012", "split_config"))

    logger = logging.getLogger("detectron2.data.datasets.Sim10k")
    logger.info(
        "Preprocessing Sim10k object detection annotations of {} set...".format(split))

    # to gather the data, we first obtain the list of image IDs belonging
    # to the current split from the respective split config file
    # e.g. train.txt or val.txt in split_config
    with open(os.path.join(data_dir, "VOC2012", "split_config", f"{split}.txt")) as f:
        image_ids = f.read().splitlines()

    # sample a certain number of images
    if cfg.FOOC.DATASETS.SIM10K.SUBSAMPLE == 0:
        err_msg = "You are trying to draw a subsample of length zero."
        raise ValueError(err_msg)
    elif cfg.FOOC.DATASETS.SIM10K.SUBSAMPLE != -1:
        image_ids = random.sample(
            image_ids, cfg.FOOC.DATASETS.SIM10K.SUBSAMPLE)
    assert len(image_ids)

    # parse the gathered files into detectron2 data dict
    domain = cfg.FOOC.DATASETS.SIM10K.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.SIM10K.LABEL_SPACE)

    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    records = pool.map(
        functools.partial(
            sim10k_sample_to_dict,
            data_dir=data_dir,
            is_train=True,
            domain=domain,
            label_space=label_space),
        image_ids,
    )
    logger.info("Loaded {} images from {}".format(len(records), data_dir))
    pool.close()

    return records


def sim10k_sample_to_dict(image_id, data_dir, is_train, domain, label_space):
    """
    Parse SIM10k annotation files to a instance detection dataset dict.

    Args:
        image_id(str): image id
        data_dir(str): directory for storing images and labels
        is_train(bool): whether to prepare the dataset for training
        domain(str): domain of the image 
        label_space(cs_labels): SIM10k label space

    Returns:
        A dict of 1 image in Detectron2 Dataset format.
    """
    # determine the storage path of the image and its label based on its ID
    image_path = os.path.join(
        data_dir, "VOC2012", "JPEGImages", f"{image_id}.jpg")
    annos_path = os.path.join(
        data_dir, "VOC2012", "Annotations", f"{image_id}.xml")
    assert os.path.exists(image_path)
    assert os.path.exists(annos_path)

    # load the image to get its size
    image = detection_utils.read_image(image_path, format="BGR")

    # record the basic information of the image
    record = {
        "file_name": image_path,
        "image_id": image_id,
        "height": image.shape[0],
        "width": image.shape[1],
    }

    del image

    # record the domain information of the image
    if is_train:
        if domain == "source":
            record["domain"] = np.zeros([1], dtype=np.float32)
        elif domain == "target":
            record["domain"] = np.ones([1], dtype=np.float32)
        elif domain is not None:
            raise ValueError(
                "The dataset domain can only be either \"source\", \"target\" or \"None.\""
            )

    # read all instances in the image
    tree = ET.parse(annos_path)

    annos = []
    for obj in tree.findall("object"):
        type = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox = [
            float(bndbox.find("xmin").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymax").text),
        ]

        if label_space.__name__ not in ["Sim10k", "Sim10kCityscapesCommon"]:
            err_msg = "You are trying to map classes of {} label space into Sim10k format.".format(
                str(label_space.__name__))
            raise ValueError(err_msg)
        else:
            if label_space.__name__ == "Sim10kCityscapesCommon":
                type = label_space.from_sim10k[type]
            label = label_space.name2label[type]

        # remove instances of irrelevant categories
        if not label.hasInstances or label.ignoreInEval:
            continue

        # make the ids of the remaining categories continuous
        contiguous_id_dict = {l.id: idx for idx,
                              l in enumerate(label_space.instance_labels)}

        # only 2D information will be loaded
        anno = {
            "iscrowd": False,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": contiguous_id_dict[label.id],
        }

        annos.append(anno)

    record["annotations"] = annos
    return record

# ==================== KITTI ==================== #
def register_kitti(cfg, root_dir="datasets"):
    """
    Register KITTI dataset in the standard Detectron2 annotation format for
    2D instance detection.

    Args:
        cfg(config): global configs of Detectron2
        root_dir (str or path-like): directory which contains all the data.
    """
    # set the data directory and basic configs of the KITTI dataset
    data_dir = os.path.join(root_dir, "Kitti")
    domain = cfg.FOOC.DATASETS.KITTI.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.KITTI.LABEL_SPACE)

    # load valid class names of labels
    thing_classes = [label.name for label in label_space.labels
                     if label.hasInstances and not label.ignoreInEval]
    stuff_classes = [label.name for label in label_space.labels
                     if not label.hasInstances]

    # split the data set into the training set and the validation set according to the preset split method, and then register them
    for split in ["train", "val"]:
        DatasetCatalog.register(
            "Kitti_"+split,
            lambda data_dir=data_dir, split=split, config=cfg:
                load_kitti_samples(data_dir, split, config)
        )

        MetadataCatalog.get("Kitti_"+split).set(
            data_dir=data_dir,
            evaluator_type="coco",
            domain=domain,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
        )


def load_kitti_samples(data_dir, split, cfg):
    """
    Select appropriate samples according to the preset split method to create
    a dict of the entire dataset

    Args:
        data_dir(str): directory for storing images and labels
        split(str): dataset type ("train" or "val")
        cfg(config): global configs of Detectron2

    Returns:
        A dict list of all image within a dataset in Detectron2 Dataset format.
    """
    # verify the validity of the directories
    assert os.path.exists(os.path.join(data_dir, "training", "image_2"))
    assert os.path.exists(os.path.join(data_dir, "training", "label_2"))
    assert os.path.exists(os.path.join(data_dir, "training", "split_config"))

    logger = logging.getLogger("detectron2.data.datasets.Kitti")
    logger.info(
        "Preprocessing kitti object detection annotations of {} set...".format(split))

    # to gather the data, we first obtain the list of image IDs belonging
    # to the current split from the respective split config file
    # e.g. train.txt or val.txt in split_config
    with open(os.path.join(data_dir, "training", "split_config", f"{split}.txt")) as f:
        image_ids = f.read().splitlines()

    # sample a certain number of images
    if cfg.FOOC.DATASETS.KITTI.SUBSAMPLE == 0:
        err_msg = "You are trying to draw a subsample of length zero."
        raise ValueError(err_msg)
    elif cfg.FOOC.DATASETS.KITTI.SUBSAMPLE != -1:
        image_ids = random.sample(image_ids, cfg.FOOC.DATASETS.KITTI.SUBSAMPLE)
    assert len(image_ids)

    # parse the gathered files into detectron2 data dict
    domain = cfg.FOOC.DATASETS.KITTI.DOMAIN
    label_space = get_label_space(cfg.FOOC.DATASETS.KITTI.LABEL_SPACE)

    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    records = pool.map(
        functools.partial(
            kitti_sample_to_dict,
            data_dir=data_dir,
            is_train=True,
            domain=domain,
            label_space=label_space),
        image_ids,
    )
    logger.info("Loaded {} images from {}".format(len(records), data_dir))
    pool.close()

    return records


def kitti_sample_to_dict(image_id, data_dir, is_train, domain, label_space):
    """
    Parse KITTI annotation files to a instance detection dataset dict.

    Args:
        image_id(str): image id
        data_dir(str): directory for storing images and labels
        is_train(bool): whether to prepare the dataset for training
        domain(str): domain of the image 
        label_space(cs_labels): KITTI label space

    Returns:
        A dict of 1 image in Detectron2 Dataset format.
    """
    # determine the storage path of the image and its label based on its ID
    image_path = os.path.join(data_dir, "training",
                              "image_2", f"{image_id}.png")
    annos_path = os.path.join(data_dir, "training",
                              "label_2", f"{image_id}.txt")
    assert os.path.exists(image_path)
    assert os.path.exists(annos_path)

    # load the image to get its size
    image = detection_utils.read_image(image_path, format="BGR")

    # record the basic information of the image
    record = {
        "file_name": image_path,
        "image_id": image_id,
        "height": image.shape[0],
        "width": image.shape[1],
    }

    del image

    # record the domain information of the image
    if is_train:
        if domain == "source":
            record["domain"] = np.zeros([1], dtype=np.float32)
        elif domain == "target":
            record["domain"] = np.ones([1], dtype=np.float32)
        elif domain is not None:
            raise ValueError(
                "The dataset domain can only be either \"source\", \"target\" or \"None.\""
            )

    # read all instances in the image
    with open(annos_path, 'r') as f:
        # Values    Name      Description
        # ----------------------------------------------------------------------------
        #    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
        #                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
        #                      'Misc' or 'DontCare'
        #    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
        #                      truncated refers to the object leaving image boundaries
        #    1    occluded     Integer (0,1,2,3) indicating occlusion state:
        #                      0 = fully visible, 1 = partly occluded
        #                      2 = largely occluded, 3 = unknown
        #    1    alpha        Observation angle of object, ranging [-pi..pi]
        #    4    bbox         2D bounding box of object in the image (0-based index):
        #                      contains left, top, right, bottom pixel coordinates
        #    3    dimensions   3D object dimensions: height, width, length (in meters)
        #    3    location     3D object location x,y,z in camera coordinates (in meters)
        #    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        #    1    score        Only for results: Float, indicating confidence in
        #                      detection, needed for p/r curves, higher is better.
        instances = f.readlines()

    # record the category and 2D information of each valid instance
    annos = []
    for instance in instances:
        instance = instance.strip().split(' ')

        type = str(instance[0])
        # truncated=float(instance[1])
        # occluded=int(instance[2])
        # alpha=float(instance[3])
        bbox = list(map(float, instance[4:8]))
        # dimensions=list(map(float, instance[8:11]))
        # location=list(map(float, instance[11:14]))
        # rotation_y=float(instance[14])
        # if len(instance)>15:
        #     score=float(instance[15])

        if label_space.__name__ not in ["Kitti", "KittiCityscapesCommon"]:
            err_msg = "You are trying to map classes of {} label space into Kitti format.".format(
                str(label_space.__name__))
            raise ValueError(err_msg)
        else:
            if label_space.__name__ == "KittiCityscapesCommon":
                type = label_space.from_kitti[type]
            label = label_space.name2label[type]

        # remove instances of irrelevant categories
        if not label.hasInstances or label.ignoreInEval:
            continue

        # make the ids of the remaining categories continuous
        contiguous_id_dict = {l.id: idx for idx,
                              l in enumerate(label_space.instance_labels)}

        # only 2D information will be loaded
        anno = {
            "iscrowd": False,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": contiguous_id_dict[label.id],
        }

        annos.append(anno)

    record["annotations"] = annos
    return record
