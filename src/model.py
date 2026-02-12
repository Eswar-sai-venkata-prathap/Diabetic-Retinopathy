from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2


def build_model(input_shape=(224, 224, 3)):
    """
    Binary Diabetic Retinopathy Detection Model
    Output:
        0 → No DR
        1 → DR Present
    """

    # =========================
    # Base Model (Transfer Learning)
    # =========================
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze majority of layers
    for layer in base_model.layers:
        layer.trainable = False

    # Fine-tune last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # =========================
    # Classification Head
    # =========================
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)

    x = Dropout(0.5)(x)

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)

    x = Dropout(0.3)(x)

    # 🔥 Binary Output
    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
