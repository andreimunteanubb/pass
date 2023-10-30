from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers
import tensorflow as tf


def train_model(model, train_gen, valid_gen):
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=40,
        decay_rate=0.96,
        staircase=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  # optimizer=Adam(learning_rate=0.00001,decay = 10e-5),
                  metrics=['accuracy'])
    # option = run_opts)

    """3-2: Model training"""

    history = model.fit(x=train_gen,
                        epochs=30,
                        validation_data=valid_gen,
                        steps_per_epoch=None,
                        workers=2
                        )
