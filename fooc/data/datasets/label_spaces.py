from cityscapesscripts.helpers.labels import Label, labels as cs_labels

# **** single dataset label spaces **** #
class Cityscapes:
    # labels = [
    #     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    #     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    #     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    #     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    #     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    #     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    #     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    #     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    #     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    #     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    #     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    #     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    #     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    #     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    #     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    #     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    #     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    #     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    #     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    #     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    #     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    #     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    #     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    #     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    #     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    #     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    #     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    #     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    #     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    #     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    #     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    #     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    # ]
    labels = cs_labels
    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]
    box3d_labels = [l for l in instance_labels if l.name !=
                    "rider" and l.name != "person"]


class Sim10k:
    labels = [
        #       name              id  trainId   category        catId   hasInstances    ignoreInEval    color               # noqa
        Label('car',              24, 19,       'vehicle',      7,      1,              0,              (0, 0, 142)),       # noqa
        Label('motorbike',        23, 18,       'vehicle',      7,      1,              0,              (0, 0, 230)),       # noqa
        Label('person',           20, 17,       'human',        6,      1,              0,              (220, 20, 60)),     # noqa
    ]

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


class Kitti:
    """
    Car, Pedestrian, Cyclist, Dont Care
    1. Pedestrian and Person_sitting are both person in cityscapes terms
    2. Cyclist is the union of rider and bycicle in cityscapes termss
    """
    labels = [
        #       name              id  trainId   category        catId   hasInstances    ignoreInEval    color               # noqa
        Label('DontCare',         0,  255,      'void',         0,      0,              1,              (0, 0, 0)),         # noqa
        Label('Misc',             1,  255,      'void',         0,      0,              1,              (0, 0, 0)),         # noqa
        Label('Car',              2,  10,       'vehicle',      7,      1,              0,              (255, 127, 80)),    # noqa
        Label('Van',              3,  11,       'vehicle',      7,      1,              1,              (0, 139, 139)),     # noqa
        Label('Truck',            4,  12,       'vehicle',      7,      1,              1,              (160, 60, 60)),     # noqa
        Label('Pedestrian',       5,  11,       'human',        6,      1,              0,              (220, 20, 60)),     # noqa
        Label('Person_sitting',   6,  11,       'human',        6,      1,              0,              (220, 20, 60)),     # noqa
        Label('Cyclist',          7,  18,       'human',        6,      1,              0,              (119, 11, 32)),     # noqa
        Label('Tram',             8,  255,      'vehicle',      7,      1,              1,              (0, 80, 100)),      # noqa
    ]

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


class Vkitti2:
    labels = [
        #       name              id  trainId   category        catId   hasInstances    ignoreInEval    color               # noqa
        Label('unlabeled',        0,  255,      'void',         0,      0,              1,              (0, 0, 0)),         # noqa
        Label('misc',             1,  255,      'void',         0,      0,              1,              (80, 80, 80)),      # noqa What even is misc?
        Label('sky',              2,  0,        'sky',          5,      0,              0,              (90, 200, 255)),    # noqa
        Label('road',             4,  1,        'flat',         1,      0,              0,              (100, 60, 100)),    # noqa
        Label('terrain',          5,  2,        'nature',       4,      0,              0,              (210, 0, 200)),     # noqa
        Label('tree',             6,  3,        'nature',       4,      0,              0,              (0, 199, 0)),       # noqa
        Label('vegetation',       7,  4,        'nature',       4,      0,              0,              (90, 240, 0)),      # noqa
        Label('building',         8,  5,        'construction', 2,      0,              0,              (140, 140, 140)),   # noqa
        Label('guard rail',       9,  6,        'construction', 2,      0,              0,              (250, 100, 255)),   # noqa
        Label('trafficlight',     10, 7,        'object',       3,      0,              0,              (200, 200, 0)),     # noqa
        Label('trafficsign',      11, 8,        'object',       3,      0,              0,              (255, 255, 0)),     # noqa
        Label('pole',             12, 9,        'object',       3,      0,              0,              (250, 130, 0)),     # noqa
        Label('car',              13, 10,       'vehicle',      7,      1,              0,              (255, 127, 80)),    # noqa
        Label('van',              14, 11,       'vehicle',      7,      1,              0,              (0, 139, 139)),     # noqa
        Label('truck',            15, 12,       'vehicle',      7,      1,              0,              (160, 60, 60)),     # noqa
    ]

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


class CSCarsOnlyTest:
    labels = [
        #       name            id  trainId category    catId   hasInstances    ignoreInEval    color           # noqa
        Label('unlabeled',      0,  255,    'void',     0,      False,          True,           (0, 0, 0)),     # noqa
        Label('car',            26, 13,     'vehicle',  7,      True,           False,          (0, 0, 142)),   # noqa
    ]

    from_cs = {
        'person':           'unlabeled',
        # The rider class does not exist in Viper; All riders are considered persons and unlike
        # synscapes there is no way to obtain rider masks by diffing the instance and segmentation
        # ground truths. To resolve this ambiguity we remap rider to person, thus dropping this
        # distinction in the common label space.
        'rider':            'unlabeled',
        'car':              'car',
        'truck':            'unlabeled',
        'bus':              'unlabeled',
        # caravan does not exist in Viper and is treated as unlabeled
        'caravan':          'unlabeled',
        'trailer':          'unlabeled',
        # train does not exist in Viper and is treated as unlabeled
        'train':            'unlabeled',
        'motorcycle':       'unlabeled',
        # bicycle does not exist in Viper and is treated as unlabeled
        'bicycle':          'unlabeled',
        'tunnel':           'unlabeled',
    }

    for label in Cityscapes.labels:
        if label.name not in from_cs:
            from_cs.update({label.name: 'unlabeled'})

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


# **** common label spaces **** #
class Sim10kCityscapesCommon:
    """"""
    labels = cs_labels

    from_cs = {
        'person':           'unlabeled',
        'rider':            'unlabeled',
        'car':              'car',
        'truck':            'unlabeled',
        'bus':              'unlabeled',
        'caravan':          'unlabeled',
        'trailer':          'unlabeled',
        'train':            'unlabeled',
        'motorcycle':       'unlabeled',
        'bicycle':          'unlabeled',
        'tunnel':           'unlabeled',
    }

    from_sim10k = {
        'motorbike':        'unlabeled',
        'person':           'unlabeled',
        'car':              'car',
    }

    for label in Cityscapes.labels:
        if label.name not in from_cs:
            from_cs.update({label.name: 'unlabeled'})

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


class KittiCityscapesCommon:
    """"""
    labels = cs_labels

    from_cs = {
        'person':           'unlabeled',
        'rider':            'unlabeled',
        'car':              'car',
        'truck':            'truck',
        'bus':              'unlabeled',
        'caravan':          'unlabeled',
        'trailer':          'unlabeled',
        'train':            'unlabeled',
        'motorcycle':       'unlabeled',
        'bicycle':          'unlabeled',
        'tunnel':           'unlabeled',
    }

    from_kitti = {
        'DontCare':         'unlabeled',
        'Misc':             'unlabeled',
        'Car':              'car',
        'Van':              'car',
        'Truck':            'truck',
        'Pedestrian':       'person',
        'Person_sitting':   'person',
        'Cyclist':          'rider',
        'Tram':             'train',
    }

    for label in Cityscapes.labels:
        if label.name not in from_cs:
            from_cs.update({label.name: 'unlabeled'})

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


class Vkitti2CityscapesCommon:
    labels = [
        #       name              id  trainId   category        catId   hasInstances    ignoreInEval    color               # noqa
        Label('unlabeled',        0,  255,      'void',         0,      0,              1,              (0, 0, 0)),         # noqa
        Label('sky',              2,  0,        'sky',          5,      0,              0,              (90, 200, 255)),    # noqa
        Label('road',             4,  1,        'flat',         1,      0,              0,              (100, 60, 100)),    # noqa
        Label('terrain',          5,  2,        'nature',       4,      0,              0,              (210, 0, 200)),     # noqa
        Label('vegetation',       7,  4,        'nature',       4,      0,              0,              (90, 240, 0)),      # noqa
        Label('building',         8,  5,        'construction', 2,      0,              0,              (140, 140, 140)),   # noqa
        Label('guard rail',       9,  6,        'construction', 2,      0,              0,              (250, 100, 255)),   # noqa
        Label('trafficlight',     10, 7,        'object',       3,      0,              0,              (200, 200, 0)),     # noqa
        Label('trafficsign',      11, 8,        'object',       3,      0,              0,              (255, 255, 0)),     # noqa
        Label('pole',             12, 9,        'object',       3,      0,              0,              (250, 130, 0)),     # noqa
        Label('car',              13, 10,       'vehicle',      7,      1,              0,              (255, 127, 80)),    # noqa
        Label('truck',            15, 12,       'vehicle',      7,      1,              0,              (160, 60, 60)),     # noqa
    ]

    # maps Vkitti2 labels into common space with cityscapes by label name
    from_vkitti2 = {
        'unlabeled': 'unlabeled',
        'misc': 'unlabeled',  # TODO: check out what this class actually is
        'sky': 'sky',
        'road': 'road',
        'terrain': 'terrain',
        'tree': 'vegetation',
        'vegetation': 'vegetation',
        'building': 'building',
        'guard rail': 'guard rail',
        'trafficlight': 'trafficlight',
        'trafficsign': 'trafficsign',
        'pole': 'pole',
        'car': 'car',
        'van': 'car',  # vans are cars in cityscapes so we map them there
        'truck': 'truck',
    }

    # maps cityscapes labels into common space with Vkitti2 by name
    # currently only maps thing classes
    from_cs = {
        'person':           'unlabeled',  # does not exist in vkitti2
        'rider':            'unlabeled',  # does not exist in vkitti2
        'car':              'car',
        'truck':            'truck',
        'bus':              'unlabeled',  # does not exist in vkitti2
        'caravan':          'unlabeled',  # does not exist in vkitti2
        'trailer':          'unlabeled',  # does not exist in vkitti2
        'train':            'unlabeled',  # does not exist in vkitti2
        'motorcycle':       'unlabeled',  # does not exist in vkitti2
        'bicycle':          'unlabeled',  # does not exist in vkitti2
    }

    # dictionaries for convenient conversion
    # name to label object
    name2label = {label.name: label for label in labels}
    # id to label object
    id2label = {label.id: label for label in labels}
    # trainId to label object
    trainId2label = {label.trainId: label for label in reversed(labels)}
    # get list of all labels relevant for detection tasks
    instance_labels = [
        l for l in labels if l.hasInstances and not l.ignoreInEval]


def get_label_space(name):
    """"""
    if name == "Cityscapes":
        return Cityscapes
    elif name == "Kitti":
        return Kitti
    elif name == "Sim10k":
        return Sim10k
    elif name == "Vkitti2":
        return Vkitti2
    elif name == "Sim10kCityscapesCommon":
        return Sim10kCityscapesCommon
    elif name == "KittiCityscapesCommon":
        return KittiCityscapesCommon
    elif name == "Vkitti2CityscapesCommon":
        return Vkitti2CityscapesCommon
    elif name == "CSCarsOnlyTest":
        return CSCarsOnlyTest
    else:
        err_msg = "Unrecognized label space {}.".format(str(name))
        raise ValueError(err_msg)
