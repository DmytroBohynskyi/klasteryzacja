import keras
import numpy as np
import susi
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.utils import to_categorical

from scrips.Plot import plot_k_means

COLOR = ["#ffce00", "#5f720f", "#6c235f", "#69173a", "#0a70cf", "#89ff33", "#2f82d5", "#d47332", "#a8255d", "#653244",
         "#7a1717", "#336f3b", "#006de1", "#1f0a1b", "#751a41", ]


class AI:

    def __init__(self):
        self.data = None  # numpy array of data points
        self.data_type = None  # point type
        self.type_num = None  # number of the type
        self.x = None  # data for machine learning exactly less than one
        self.max_value = None  # max value
        # Split arrays or matrices into random train and test subsets
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.model = None  # model of machine learning
        self.som = None  # object of susi class ()
        self.model_path = None  # madel path

    def read_db(self, path: str) -> None:
        """
        This function opens csv file with data points and data type
        :param path: file position
        :return: None
        """
        # load data with c
        load_data = loadtxt(path, delimiter=',')
        self.data = load_data[:, :-1]  # numpy array of data points
        self.data_type = load_data[:, -1]  # point type

        # creat data base for machine learning
        self.type_num = int(np.amax(self.data_type)) + 1
        self.x = self.data / self.type_num  # data for machine learning exactly less than one
        self.model_path = f"models/{self.data.shape[1]}_{self.type_num}.h5"

    def train_test(self):
        """
        Split arrays or matrices into random train and test subsets
        :return: None
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.data_type, random_state=5,
                                                                                stratify=self.data_type)

    def machine_learning(self, epochs: int = 50):
        height, width = self.data.shape

        self.train_test()

        if width == 2:
            input_layer = Input(shape=(width,), dtype='float32', name='input')
            output_layer = Dense(self.type_num, activation='softmax', name='output')(input_layer)
        elif width == 9:
            input_layer = Input(shape=(width,), dtype='float32', name='input')
            hidden_layer = Dense(16, activation='relu', name='hidden')(input_layer)
            output_layer = Dense(self.type_num, activation='softmax', name='output')(hidden_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name='artifical_neutral_network')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        y_bin = to_categorical(self.y_train, num_classes=self.type_num, dtype='int')

        model.fit(self.x_train, y_bin, batch_size=1, epochs=epochs, verbose=1)

        model.save(self.model_path)

    def classification(self, algorithm="keras"):
        """
          algorithm:    "som" - self-organizing feature map algorithm function,
                        "keras" - neural network algorithm
          :return: object of susi class
          """
        self.train_test()

        if algorithm.upper() == "KERAS":
            model = load_model(self.model_path)
            model = model.predict(self.x)
            self.data_type = np.int8([np.argmax(el) for el in model])
        elif algorithm.upper() == "SOM":
            self.som = susi.SOMClassifier()
            self.som.fit(self.x_train, self.y_train)
            self.data_type = np.int8(self.som.predict(self.x))

    def plot(self, algorithm="keras"):
        centroids = self.get_centroids()
        centroids = np.array(centroids)
        plot_k_means(self.data, centroids, color=self.data_type, fig_type=algorithm)

    def get_centroids(self):
        centroids = np.zeros((self.type_num, self.data.shape[1]))
        for n, _ in enumerate(centroids):
            data_type_index = np.where(self.data_type == n)  # index value for n centroid
            if len(data_type_index[0]) != 0:  # next if n centroid dont have value
                centroids[n, :] = np.average(self.data[data_type_index], axis=0)
        return centroids
