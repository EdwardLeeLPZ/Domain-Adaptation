from torchvision.models.vgg import vgg16

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class VGG16(Backbone):
    """
    Args:
        out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        pretrained(bool): flag whether imagenet weights are used for weight initialization (True) or not
    """
    def __init__(self, out_features=None):
        super(VGG16, self).__init__()
        VGG16_layer_names = [
            "conv1_1", "relu1_1",
            "conv1_2", "relu1_2",
            "pool1",
            "conv2_1", "relu2_1",
            "conv2_2", "relu2_2",
            "pool2",
            "conv3_1", "relu3_1",
            "conv3_2", "relu3_2",
            "conv3_3", "relu3_3",
            "pool3",
            "conv4_1", "relu4_1",
            "conv4_2", "relu4_2",
            "conv4_3", "relu4_3",
            "pool4",
            "conv5_1", "relu5_1",
            "conv5_2", "relu5_2",
            "conv5_3", "relu5_3",
            "pool5",
        ]
        vgg = vgg16(pretrained=True)
        layers = list(vgg.features._modules.values())
        assert len(VGG16_layer_names) == len(layers)
        self.layers = list(zip(VGG16_layer_names, layers))

        self._out_features = out_features
        if not self._out_features:
            self._out_features = [self.layers[-1][0]]
        for out_feature in self._out_features:
            assert out_feature in VGG16_layer_names, out_feature
        
        # add out_feature_channels and out_feature_strides
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        stride = 1
        out_channels = 0
        out_features = set(self._out_features)
        for name, layer in self.layers:
            if len(out_features) == 0:
                break

            self.add_module(name, layer)

            # update out channels when the attribute is available (ConvLayers) because otherwise it doesn't change
            if hasattr(layer, "out_channels"):
                out_channels = layer.out_channels
            # update stride when the attribute is available (ConvLayers and MaxPool) because otherwise it doesn't change
            if hasattr(layer, "stride"):
                stride_multiplier = layer.stride
                if isinstance(stride_multiplier, tuple):
                    # Note: VGG only has symmetric strides
                    stride_multiplier = stride_multiplier[0]
                assert isinstance(stride_multiplier, int), type(stride_multiplier)
                stride *= stride_multiplier

            if name in self._out_features:
                self._out_feature_channels[name] = out_channels
                self._out_feature_strides[name] = stride

                out_features.remove(name)
    
    def forward(self, x):
        outputs = {}
        for name, layer in self.layers:
            x = layer(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register
def build_vgg_backbone(cfg, input_shape):
    """
    Create a VGG16 instance from config.
    Returns:
        VGG16: a :class:`VGG16` instance.
    """
    out_features = cfg.MODEL.VGG.OUT_FEATURES
    return VGG16(out_features=out_features)