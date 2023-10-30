from tensorflow.keras.preprocessing.image import ImageDataGenerator


def apply_augmentation(train_df, valid_df, test_df):
    BATCH_SIZE = 32
    IMG_SHAPE = (224, 224, 3)
    IMG_SIZE = (224, 224)

    gen = ImageDataGenerator(rescale=1. / 255,
                             vertical_flip=True,
                             horizontal_flip=True)
    # rotation_range=10)

    gen2 = ImageDataGenerator(rescale=1. / 255)

    train_gen = gen.flow_from_dataframe(train_df,
                                        x_col='filenames',
                                        y_col='labels',
                                        target_size=IMG_SIZE,
                                        class_mode='categorical',
                                        color_mode='rgb',
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,
                                        seed=88
                                        )

    valid_gen = gen2.flow_from_dataframe(valid_df,
                                         x_col='filenames',
                                         y_col='labels',
                                         target_size=IMG_SIZE,
                                         class_mode='categorical',
                                         color_mode='rgb',
                                         shuffle=True,
                                         batch_size=BATCH_SIZE,
                                         seed=88
                                         )
    test_gen = gen2.flow_from_dataframe(test_df,
                                        x_col='filenames',
                                        y_col='labels',
                                        target_size=IMG_SIZE,
                                        class_mode='categorical',
                                        color_mode='rgb',
                                        shuffle=True,
                                        batch_size=325,
                                        seed=88
                                        )