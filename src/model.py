from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, SpatialDropout2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(224, 224, 3), num_classes=5, learning_rate=1e-3):
    # MobileNetV2
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze first 70%
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * 0.70)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
        
    x = base_model.output
    # SpatialDropout2D requires 4D tensor. base_model output is 4D (7x7x1280 for 224 input)
    x = SpatialDropout2D(0.3)(x)
    x = GlobalAveragePooling2D()(x)
    
    # Dense Head with L2
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
