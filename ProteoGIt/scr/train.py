import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

# ================= CONFIGURAÇÕES GENÉRICAS E RELATIVAS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'document_classifier_v11.h5')
JSON_CLASSES_PATH = os.path.join(BASE_DIR, 'config', 'class_indices.json')

IMG_HEIGHT, IMG_WIDTH = 240, 240 
BATCH_SIZE = 16 
EPOCHS_FASE_1 = 20  
EPOCHS_FASE_2 = 40  

class RealTimeLogger(Callback):
    """Callback para monitorar e registrar métricas durante o treinamento."""
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.last_log_time = time.time()
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        agora = time.time()
        if agora - self.last_log_time >= 60:
            tempo_decorrido = agora - self.start_time
            print(f"\n[SISTEMA {datetime.datetime.now().strftime('%H:%M:%S')}]")
            print(f" > Status: Processando Época {self.params['epochs']} - Batch {batch}")
            print(f" > Acurácia Atual: {logs.get('accuracy', 0):.4f} | Perda: {logs.get('loss', 0):.4f}")
            print(f" > Tempo de execução: {str(datetime.timedelta(seconds=int(tempo_decorrido)))}")
            self.last_log_time = agora

def prepare_dataset():
    """Gera dados com aumento de imagem (Data Augmentation) e validação split."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        zoom_range=0.05,
        fill_mode='constant',
        cval=0,
        validation_split=0.2 
    )

    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print("\n[INFO] Carregando geradores de dados...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR, 
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, 
        class_mode='categorical', 
        subset='training', 
        shuffle=True
    )

    validation_generator = valid_datagen.flow_from_directory(
        DATASET_DIR, 
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, 
        class_mode='categorical', 
        subset='validation', 
        shuffle=False
    )

    return train_generator, validation_generator, train_generator.class_indices

def build_model(num_classes):
    """Constrói o modelo usando EfficientNetB0 com ativação Swish."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='swish')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model, base_model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERRO] O diretório do dataset não existe: {DATASET_DIR}")
        return

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(JSON_CLASSES_PATH), exist_ok=True)

    train_gen, val_gen, class_indices = prepare_dataset()
    
    with open(JSON_CLASSES_PATH, 'w') as f:
        json.dump(class_indices, f, indent=4)

    model, base_model = build_model(len(class_indices))

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)

    print("\n--- FASE 1: Extração de Features ---")
    model.fit(
        train_gen, 
        epochs=EPOCHS_FASE_1, 
        validation_data=val_gen, 
        callbacks=[checkpoint, early_stop, reduce_lr, RealTimeLogger(EPOCHS_FASE_1)]
    )

    print("\n--- FASE 2: Fine-Tuning ---")
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    model.fit(
        train_gen, 
        epochs=EPOCHS_FASE_2, 
        validation_data=val_gen, 
        callbacks=[checkpoint, early_stop, reduce_lr, RealTimeLogger(EPOCHS_FASE_2)]
    )

    print(f"\nTreinamento finalizado. Modelo salvo em: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()