import os.path
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
from keras import backend as keras_backend
from keras_applications.resnet import ResNet101
from keras.optimizers import Adam
import json

from pylmnn.dataset.load_dataset import _dataset_verification, _dataset_split_fewshot


def sort_and_dedup(input_list: list):
    """Sorts a list alphabetically and remove  duplicates. Return tuple"""
    seen = set()
    seen_add = seen.add
    input = sorted(input_list)  # sort
    uniques = [x for x in input if not (x in seen or seen_add(x))]
    output = tuple(uniques)

    return output

class ConfigParam:
    def __init__(self, name, default=None, doc=None):
        self.name = name
        self.default = default
        self.__doc__ = doc

    def __get__(self, obj, type=None):
        if obj is None:
            return self

        if self.name not in obj.params:
            return self.default

        return obj.params[self.name]

    def __set__(self, obj, value):
        raise AttributeError('Config parameter %s is readonly' % self.name)

    def __delete__(self, obj):
        raise AttributeError('Config parameter %s is readonly' % self.name)

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.name)


class Config:
    def __init__(self, overrides=None):
        self.params = {}

        if overrides is not None:
            for k, v in overrides.items():
                # Ignore the override if it's not overriding anything
                if not hasattr(self, k):
                    continue

                self.params[k] = v

    def __iter__(self):
        genealogy = [self.__class__]

        # Iterate through all ConfigParam in current and parent classes
        while len(genealogy) > 0:
            cls = genealogy.pop()
            for v in cls.__dict__.values():
                if isinstance(v, ConfigParam):
                    yield v.name, getattr(self, v.name)
            genealogy += list(cls.__bases__)


class ClassConfig(Config):
    """ResNet101 Parameters
    """
    model_name = ConfigParam(
        'model_name', 'resnet101.hdf5',
        'Name of saved Classificaition Model'
    )

    schema_name = ConfigParam(
        'schema_name', 'schema.json',
        'Name of schema for trained Classification Model'
        '[class names and indices]'
    )

    metrics_name = ConfigParam(
        'metrics_name', 'metrics.json',
        'Name of metrics file for trained Classification Model [dev only]'
    )

    num_rows = ConfigParam(
        'num_rows', 128,
        'Number of rows in resized image for Classification')

    num_cols = ConfigParam(
        'num_cols', 128,
        'Number of columns in resized image for Classification')

    max_size = ConfigParam(
        'max_size', 448,
        'max number of rows/columns in resized image for Classification')

    num_channels = ConfigParam(
        'num_channels', 3,
        'Number of channels (e.g. RGB) assumed for input images'
    )

    initial_weights = ConfigParam(
        'initial_weights', 'imagenet',
        'Dataset used for initial pretrained weights for Classification Model'
    )

    channel_order = ConfigParam(
        'channel_order', 'RGB',
        'Channel Order used for Classification Model. RGB or BGR'
    )

    pixel_means = ConfigParam(
        'pixel_means', (103.939, 116.779, 123.68),
        'Pixel Means used for pretrained IMAGENET weights. BGR'
    )

    training_batch_size = ConfigParam(
        'training_batch_size', 16,
        'Maximum batch size for Classification Training'
    )

    prediction_batch_size = ConfigParam(
        'prediction_batch_size', 16,
        'Maximum batch size for Classification Predictions'
    )

    num_epochs = ConfigParam(
        'num_epochs', 50,
        'Maximum Number of Epochs for  Classification training'
    )

    early_stop_patience = ConfigParam(
        'early_stop_patience', 5,
        'Number of epochs early stopping will run after max/min is achieved'
    )

    early_stop_monitor = ConfigParam(
        'early_stop_monitor', 'acc',
        'Number of epochs early stopping will run after max/min is achieved'
        '["val_loss", "val_acc", "loss", "acc"]'
    )

    adam_lr = ConfigParam(
        'adam_lr', 0.0001,
        'Learning Rate for Adam Optimizer, Classification training'
    )

    adam_beta_1 = ConfigParam(
        'adam_beta_1', 0.9,
        'Regularization Beta for Adam Optimizer, Classification training'
    )


class DataGenerator(keras.utils.Sequence):
    'Generates data for batch-processing Classification Model'

    def __init__(self, dataset, labels, batch_size=16, dim=(128, 128),
                 num_channels=3, pixel_means=(103.939, 116.779, 123.68),
                 shuffle=False, collect_labels=False):
        'Initialization'
        self.shuffle = shuffle
        self.dim = dim
        self.num_channels = num_channels
        self.pixel_means = pixel_means
        self.labels = labels
        self.label_idx = dict(zip(labels, range(0, len(labels))))
        self.num_sample = 0
        self.list_IDs = []
        self.class_indexes = []
        self.class_indexes_tmp = []
        self.collect_labels = collect_labels
        self.epoch = 0

        for label in labels:
            self.num_sample += len(dataset[label])
            self.list_IDs += [[dataset[label][i], label] for i in range(len(dataset[label]))]

        if self.shuffle:
            np.random.seed(self.epoch)
            np.random.shuffle(self.list_IDs)

        self.batch_size = max(min(batch_size, self.num_sample), 1)

        self.n_classes = len(dataset.keys())

        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) // self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size
                  ]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)
        if self.collect_labels:
            self.class_indexes_tmp.extend(y)

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.epoch += 1
        if self.shuffle:
            np.random.seed(self.epoch)
            np.random.shuffle(self.indexes)
        print(len(self.class_indexes_tmp))
        self.class_indexes = self.class_indexes_tmp
        self.class_indexes_tmp = []

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples' # X :
         (n_samples, *dim, num_channels)"""

        x = np.empty((self.batch_size, self.dim[0], self.dim[1],
                      self.num_channels), dtype='float32')
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for idx, image_info in enumerate(list_ids_temp):
            image_path, label = image_info
            img = Image.open(image_path)
            img = np.array(img.resize((self.dim[0], self.dim[1])))

            x[idx,] = img
            x[idx, :, :, 0] -= self.pixel_means[0]
            x[idx, :, :, 1] -= self.pixel_means[1]
            x[idx, :, :, 2] -= self.pixel_means[2]

            y[idx] = self.label_idx[label]

        return x, y


class ClassificationTrainer:
    def __init__(self, labels, config=ClassConfig()):
        self.config = config
        self.labels = labels
        self.num_classes = len(labels)
        self._schema()

    def train_resnet101(self, training_generator, validation_generator):
        print("Training Classification Model...")

        checkpoint = ModelCheckpoint(self.config.model_name,
                                     monitor=self.config.early_stop_monitor,
                                     verbose=1, save_best_only=True)

        early_stopping = EarlyStopping(monitor=self.config.early_stop_monitor,
                                       patience=self.config.early_stop_patience,
                                       restore_best_weights=True)

        base_model = ResNet101(include_top=False,
                               weights=self.config.initial_weights,
                               backend=keras_backend,
                               layers=keras.layers,
                               models=keras.models,
                               utils=keras.utils)
        x = base_model.output
        x = GlobalAveragePooling2D(name='features')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model -
        # should be done *after* setting layers to non-trainable
        model.compile(optimizer=Adam(lr=self.config.adam_lr,
                                     beta_1=self.config.adam_beta_1),
                      loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            verbose=1, callbacks=[early_stopping, checkpoint],
            epochs=self.config.num_epochs
        )

        self._save_metrics(history.history)
        # VIP: https://github.com/keras-team/keras/issues/6462
        keras_backend.clear_session()
        return history

    def extract_feature(self, data_generator):
        model = load_model(self.config.model_name)
        features = Model(inputs=model.input,
                         outputs=model.get_layer('features').output)

        feature_vector = features.predict_generator(data_generator, verbose=1)
        keras_backend.clear_session()
        return feature_vector

    def predict(self, data_generator):
        model = load_model(self.config.model_name)

        prob = model.predict_generator(data_generator, verbose=1)
        keras_backend.clear_session()
        return prob

    def _schema(self):
        # Save schema
        ordered_labels = tuple(self.labels)
        schema = json.dumps(ordered_labels, ensure_ascii=False) \
            .encode('utf-8')
        open(self.config.schema_name, 'wb').write(schema)

    def _save_metrics(self, history: dict):
        # # # Print Metrics # # #
        train_acc = history['acc']
        val_acc = history['val_acc']
        train_loss = history['loss']
        val_loss = history['val_loss']
        print("Classification, training accuracy {}"
              .format(train_acc))
        print("Classification, validation accuracy: {}"
              .format(val_acc))
        print("Classification, train_loss: {}".format(train_loss))
        print("Classification, val_loss: {}".format(val_loss))

        # # # Save Metrics # # #
        classification_metrics = {"train_acc": train_acc,
                                  "val_acc": val_acc,
                                  "train_loss": train_loss,
                                  "val_loss": val_loss}
        metrics = json.dumps(classification_metrics, ensure_ascii=False) \
            .encode('utf-8')
        open(self.config.metrics_name, 'wb').write(metrics)


if __name__ == '__main__':
    training = False
    data_path = '/export/home/loreal_135_classification'  # '/Users/zyuan/Downloads/loreal_135_classification'
    image_dict, label2id = _dataset_verification(data_path)
    data = _dataset_split_fewshot(image_dict, 50)

    # training
    train_set = data['train']
    test_set = data['test']
    data_labels = sort_and_dedup(list(train_set.keys()))
    print(data_labels)

    config = ClassConfig()
    cls_trainer = ClassificationTrainer(labels=data_labels, config=config)

    if training:
        training_generator = DataGenerator(dataset=train_set, labels=data_labels,
                                           batch_size=config.training_batch_size,
                                           dim=(config.num_rows, config.num_cols),
                                           num_channels=config.num_channels, shuffle=True)
        validation_generator = DataGenerator(dataset=test_set, labels=data_labels,
                                             batch_size=config.training_batch_size,
                                             dim=(config.num_rows, config.num_cols),
                                             num_channels=config.num_channels, shuffle=True)
        cls_trainer.train_resnet101(training_generator=training_generator,
                                    validation_generator=validation_generator)

    feature_training_generator = DataGenerator(dataset=train_set, labels=data_labels,
                                               batch_size=config.training_batch_size,
                                               dim=(config.num_rows, config.num_cols),
                                               num_channels=config.num_channels, collect_labels=True)
    feature_validation_generator = DataGenerator(dataset=test_set, labels=data_labels,
                                                 batch_size=config.training_batch_size,
                                                 dim=(config.num_rows, config.num_cols),
                                                 num_channels=config.num_channels, collect_labels=True)
    # save embeddings
    feature_training = cls_trainer.extract_feature(data_generator=feature_training_generator)
    label_training = np.array(feature_training_generator.class_indexes)
    np.savez_compressed(os.path.join(data_path, 'em_training.npz'), X=feature_training, y=label_training)

    feature_validation = cls_trainer.extract_feature(data_generator=feature_validation_generator)
    label_validation = np.array(feature_validation_generator.class_indexes)
    np.savez_compressed(os.path.join(data_path, 'em_test.npz'), X=feature_validation, y=label_validation)

    print(feature_training.shape)
    print(len(label_training))
    print(feature_validation.shape)
    print(len(label_validation))

    # verify the accuracy in both training and validation set

    feature_training_generator.class_indexes = []
    predicted_training = cls_trainer.predict(data_generator=feature_training_generator)
    predicted_class_indices = np.argmax(predicted_training, axis=1)

    labels_training = np.array(feature_training_generator.class_indexes)
    print(np.mean(np.equal(labels_training, predicted_class_indices)))

    feature_validation_generator.class_indexes = []
    predicted_testing = cls_trainer.predict(data_generator=feature_validation_generator)
    predicted_class_indices = np.argmax(predicted_testing, axis=1)

    labels_training = np.array(feature_validation_generator.class_indexes)
    print(np.mean(np.equal(labels_training, predicted_class_indices)))




