import efficientnet.tfkeras as efn
import tensorflow as tf, tensorflow.keras.backend as K
from utils.global_var import CLASSES
# for tensorflow.keras
from classification_models.tfkeras import Classifiers
def get_model(config):

    # model = globals().get(config.MODEL.NAME)(1)
    print('model name:', config.MODEL.NAME)
    model_name = config.MODEL.NAME
    input_shape = (config.DATA.IMG_H, config.DATA.IMG_W, 3)
    pretrained_weight = config.MODEL.WEIGHT
    if pretrained_weight == 'None':
        pretrained_weight = None
    if 'EfficientNet' in model_name:
        ##keras.application
        if 'B7' in model_name:
            encoder = efn.EfficientNetB7(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B0' in model_name:
            encoder = efn.EfficientNetB0(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B1' in model_name:
            encoder = efn.EfficientNetB1(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B2' in model_name:
            encoder = efn.EfficientNetB2(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B3' in model_name:
            encoder = efn.EfficientNetB3(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B4' in model_name:
            encoder = efn.EfficientNetB4(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B5' in model_name:
            encoder = efn.EfficientNetB5(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        elif 'B6' in model_name:
            encoder = efn.EfficientNetB6(
                input_shape=input_shape,
                weights=pretrained_weight,
                include_top=False
            )
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])

    else:
        ##https://github.com/qubvel/classification_models
        #['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        # 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnet50v2', 'resnet101v2',
        # 'resnet152v2', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
        # 'densenet121', 'densenet169', 'densenet201',
        # 'inceptionresnetv2', 'inceptionv3', 'xception', 'nasnetlarge', 'nasnetmobile', 'mobilenet', 'mobilenetv2']

        base_model, preprocess_input = Classifiers.get(model_name)
        base_model = base_model(input_shape = input_shape,weights=pretrained_weight, include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

        output = tf.keras.layers.Dense(len(CLASSES), activation=config.MODEL.OUTPUT_ACTIVATION)(x)
        # if 'focal' in config.LOSS.NAME:
        #     if 'categorical_focal_loss' == config.LOSS.NAME:
        #     else:
        #         output = tf.keras.layers.Dense(len(CLASSES), activation='sigmoid')(x)
        # else:
        #     output = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)