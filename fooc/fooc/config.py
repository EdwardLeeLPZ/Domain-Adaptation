from detectron2 import model_zoo
from detectron2.config import CfgNode as CN

def add_fooc_config(cfg):
    _C = cfg

    _C.DATASETS.ROOT_DIR = "/lhome/peizhli/datasets"
    _C.DATASETS.TRAIN = ("Cityscapes_train",) # "Cityscapes_train", "FoggyCityscapes_train", "Sim10k_train" or "Kitti_train"
    _C.DATASETS.TEST = ("Cityscapes_val",) # "Cityscapes_val", "FoggyCityscapes_val", "Sim10k_val" or "Kitti_val"

    _C.DATALOADER.NUM_WORKERS = 4

    _C.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 
    # _C.MODEL.WEIGHTS = '/lhome/peizhli/projects/model_zoo/model_final_f97cb7.pkl' # original Resnet model
    # _C.MODEL.WEIGHTS = '/lhome/peizhli/projects/FOOC/baseline/cityscapes/model_final.pth'

    _C.SOLVER.IMS_PER_BATCH = 6
    _C.SOLVER.BASE_LR = 1e-4
    _C.SOLVER.MAX_ITER = 50000
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.WEIGHT_DECAY = 5e-4

    _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 8  # No. of classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    
    _C.TEST.EVAL_PERIOD = 50000
    _C.OUTPUT_DIR = '/lhome/peizhli/context-disentanglement/fooc/baseline/cityscapes'

    _C.FOOC = CN()
    _C.FOOC.NUM_DOMAINS = 2
    _C.FOOC.AUGMENTATIONS = True
    _C.FOOC.TASK = "domain_adaptation"
    _C.FOOC.BOX_EVAL = True
    _C.FOOC.BOX_EVAL_RESIZE = 1.

    _C.FOOC.SOURCE = CN()
    _C.FOOC.SOURCE.COMPUTE_DET_LOSS = True
    _C.FOOC.SOURCE.COMPUTE_INST_LOSS = True
    _C.FOOC.SOURCE.COMPUTE_3D_LOSS = True
    _C.FOOC.SOURCE.COMPUTE_3D_REPROJECTION_LOSS = False

    _C.FOOC.TARGET = CN()
    _C.FOOC.TARGET.COMPUTE_DET_LOSS = True
    _C.FOOC.TARGET.COMPUTE_INST_LOSS = False
    _C.FOOC.TARGET.COMPUTE_3D_LOSS = False
    _C.FOOC.TARGET.COMPUTE_3D_REPROJECTION_LOSS = False

    _C.FOOC.DATASETS = CN()
    _C.FOOC.DATASETS.ONLINE_LOADING = False

    _C.FOOC.DATASETS.CITYSCAPES = CN()
    _C.FOOC.DATASETS.CITYSCAPES.SUBSAMPLE = -1
    _C.FOOC.DATASETS.CITYSCAPES.DOMAIN = "source"
    _C.FOOC.DATASETS.CITYSCAPES.LABEL_SPACE = "Cityscapes"
    _C.FOOC.DATASETS.CITYSCAPES.LOAD_MASKS = True

    _C.FOOC.DATASETS.FOGGYCITYSCAPES = CN()
    _C.FOOC.DATASETS.FOGGYCITYSCAPES.SUBSAMPLE = -1
    _C.FOOC.DATASETS.FOGGYCITYSCAPES.DOMAIN = "target"
    _C.FOOC.DATASETS.FOGGYCITYSCAPES.LABEL_SPACE = "Cityscapes"
    _C.FOOC.DATASETS.FOGGYCITYSCAPES.LOAD_MASKS = True
    _C.FOOC.DATASETS.FOGGYCITYSCAPES.BETA = 0.01

    _C.FOOC.DATASETS.SIM10K = CN()
    _C.FOOC.DATASETS.SIM10K.SUBSAMPLE = -1
    _C.FOOC.DATASETS.SIM10K.DOMAIN = "target"
    _C.FOOC.DATASETS.SIM10K.LABEL_SPACE = "Sim10kCityscapesCommon" # "Sim10k" or "Sim10kCityscapesCommon" (only used for cross-domain object detection test)
    _C.FOOC.DATASETS.SIM10K.LOAD_MASKS = True

    _C.FOOC.DATASETS.KITTI = CN()
    _C.FOOC.DATASETS.KITTI.SUBSAMPLE = -1
    _C.FOOC.DATASETS.KITTI.DOMAIN = "target"
    _C.FOOC.DATASETS.KITTI.LABEL_SPACE = "KittiCityscapesCommon" # "Kitti" or "KittiCityscapesCommon" (only used for cross-domain object detection test)
    # Kitti object challenge has no instance masks
    # _C.FOOC.DATASETS.KITTI.LOAD_MASKS = True