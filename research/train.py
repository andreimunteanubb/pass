from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras import regularizers
import tensorflow as tf

from research.data_processing.data_loaders import dataset_split, read_data
from research.data_processing.data_preprocessing import apply_augmentation
from research.data_processing.dataframes_creation import create_train_dataframe, create_test_dataframe, \
    create_val_train_split
from research.models.model_creation import create_model


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
    return history


if __name__ == "__main__":
    # dataset split
    train_list, test_list = dataset_split()

    # prepare datasets
    # TODO: check if data is already ready and discard
    # read_data(train_list, "prepare_data", True)
    # read_data(test_list, "prepare_test_data")

    # create data frames
    training_df = create_train_dataframe()
    test_df = create_test_dataframe()

    train_df, valid_df = create_val_train_split(training_df)

    # apply augmentations
    train_post, valid_post, test_post = apply_augmentation(train_df, valid_df, test_df)

    # create model
    model = create_model()

    logs = train_model(model, train_post, valid_post)

