import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from typing import Optional, Dict

# -----------------------------
# Model builders
# -----------------------------
def build_lstm(window: int, n_features: int,
               lstm1: int = 64, lstm2: int = 32, dropout: float = 0.2,
               lr: float = 1e-3) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(lstm1, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(lstm2),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


# -----------------------------
# Callbacks
# -----------------------------
def default_callbacks(patience: int = 10, reduce_lr_patience: int = 5):
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=reduce_lr_patience, verbose=1)
    return [es, rlr]


# -----------------------------
# Train wrapper
# -----------------------------
def train(model: tf.keras.Model,
          X_tr, y_tr, X_val, y_val,
          epochs: int = 30, batch_size: int = 64,
          cb=None):
    if cb is None:
        cb = default_callbacks()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb,
    )
    return history