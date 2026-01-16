import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout
from flask import Flask, request
import logging
import json
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import pickle

# AWS EBS WSGI expects name application
application = app = Flask(__name__)


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConv(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = CustomConv(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = CustomConv(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.fc0 = CustomDense(512, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = CustomDense(256, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]

        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer,
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                                 trainable=True, name='b')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, x, training=None):
        input_x = x                                                     # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)                             # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)                                       # BxNx1024

        # Global features
        x = tf.reduce_max(x, axis=1)                                    # Bx1024

        # Fully-connected layers
        x = self.fc0(x, training=training)                              # Bx512
        x = self.fc1(x, training=training)                              # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)                                   # Bx1x256
        x = tf.matmul(x, self.w)                                        # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_model(bn_momentum):
    pt_cloud = Input(shape=(None, 3), dtype=tf.float32, name='pt_cloud')    # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)         # for weight-sharing of conv
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(pt_cloud_transform)
    embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_64)
    embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64)

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    embed_64_transform = tf.expand_dims(embed_64_transform, axis=2)
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_64_transform)
    hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64)
    embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_128)
    embed_1024 = tf.squeeze(embed_1024, axis=2)

    # Global feature vector (B x N x 1024 -> B x 1024)
    global_descriptor = tf.reduce_max(embed_1024, axis=1)

    # FC layers to output k scores (B x 1024 -> B x 40)
    hidden_512 = CustomDense(512, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(global_descriptor)
    hidden_512 = Dropout(rate=0.3)(hidden_512)

    hidden_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=0.3)(hidden_256)

    logits = CustomDense(40, apply_bn=False)(hidden_256)

    return Model(inputs=pt_cloud, outputs=logits)


test_model = get_model(bn_momentum=None)
# test_model.load_weights( model_path + '/iter' + str(total_steps))
# test_model.load_weights('Models/20210831_2112' + '/iter' + str(52400))
# test_model.load_weights('Models/20210908_0723' + '/iter' + str(174891))
test_model.load_weights('Models/20210917_1215' + '/iter' + str(10400))
# test_model.load_weights('Models/20210919_0923' + '/iter' + str(11000))


# Get x y z from the frame
def get_x_y_z_without_noise(frame_to_extract_x_y_z):
    sample_point = 10

    pct = frame_to_extract_x_y_z
    xyz = []

    # Calculating all the x, y and z
    for i in range(len(pct)):
        zt = pct[i][0] * np.sin(pct[i][2]) + 0.0
        xt = pct[i][0] * np.cos(pct[i][2]) * np.sin(pct[i][1])
        yt = pct[i][0] * np.cos(pct[i][2]) * np.cos(pct[i][1])
        a = [xt, yt, zt]
        xyz.append(a)

    df = pd.DataFrame(xyz)
    d_std = StandardScaler().fit_transform(df)
    db = DBSCAN(eps=2, min_samples=sample_point).fit(d_std)

    labels = db.labels_

    cluster_indexes = np.where(labels != -1)
    d_without_noise = df.iloc[cluster_indexes]

    return d_without_noise


# Removing values from list
def remove_all_by_values(list_obj, values):
    complete_list = list_obj

    for value in values:
        while value in list_obj:
            complete_list.remove(value)

    return complete_list


# detecting occupancy using v8 data
def detect_occupancy(data_to_detect_occupancy):
    data_to_detect_occupancy_without_noise = remove_all_by_values(data_to_detect_occupancy, [253, 254, 255])
    # unique_values_on_data = set(data_to_detect_occupancy_without_noise)
    # print(unique_values_on_data)
    is_occupied = len(data_to_detect_occupancy_without_noise) > 20
    return is_occupied


# Getting person height
def get_person_height(height, tilt_angle, centroidX, centroidY, centroidZ):
    rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(np.deg2rad(tilt_angle)), np.sin(np.deg2rad(tilt_angle))], [0.0, -np.sin(np.deg2rad(tilt_angle)), np.cos(np.deg2rad(tilt_angle))]])
    results = np.matmul(rotation_matrix, np.array([centroidX, centroidY, centroidZ]))
    # centroidX = results[0]
    # centroidY = results[1]
    person_height = results[2] + height
    return person_height


def z_max(data):
    frame=data
    maxElement_z = np.amax(frame[:,2])
    index_of_max_element_z = np.where(frame == np.amax(maxElement_z))
    max_pts=frame[int(index_of_max_element_z[0][0]),:]
    return max_pts


def scale_v6_data_1024(complete_data):
    frame_np = complete_data
    M = complete_data.shape[0]
    N = 1024
    mean = np.mean(frame_np, axis=0)
    sigma = np.std(frame_np, axis=0)
    frame_np = np.sqrt(N / M) * frame_np + mean - np.sqrt(N / M) * mean  # Rescale
    frame_oversampled = frame_np.tolist()
    frame_oversampled.extend([mean.tolist()] * (N - M))
    return frame_oversampled


# Detect fall Sam
@app.route('/detect_fall_batman_sam', methods=['POST'])
def detect_fall_sam():

    data = request.form

    date_time = data['date_time']
    mac_address = data['mac_address']
    stacked_v6_data = json.loads(data['stacked_v6_data'])

    # print('Room occupied')
    combined_frames = pd.DataFrame()

    for each_frame in stacked_v6_data:
        # print(each_frame)
        each_frame_x_y_z_without_noise = get_x_y_z_without_noise(each_frame)
        # print(each_frame_x_y_z_without_noise)

        concat_frames = pd.concat([combined_frames, each_frame_x_y_z_without_noise])
        combined_frames = concat_frames

    # real_time_test_data = np.load('/Users/samprabin/Documents/xealei-fall/AI/datasetsNPY/fall/test/p6-F111.npy')
    complete_data_to_test = combined_frames.to_numpy()

    ###############
    #Huy
    # print(complete_data_to_test)

    LDA_model = 'Models/knn_model.sav'
    LDA_loaded_model = pickle.load(open(LDA_model, 'rb'))
    fall_prediction = LDA_loaded_model.predict(complete_data_to_test)

    print(fall_prediction)


    ###############


    ##############
    # #Sam
    # if 1024 > len(complete_data_to_test) > 10:
    #     real_time_test_data = scale_v6_data_1024(complete_data_to_test)
    #
    #     if len(real_time_test_data) == 1024:
    #
    #         logits = test_model(np.stack([real_time_test_data]), training=False)
    #         probs = tf.math.sigmoid(logits)
    #         max_idxs = tf.math.argmax(probs, axis=1)
    #         one_hot_q = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)
    #         results = one_hot_q[0].numpy()
    #
    #         logits_numpy = logits[0].numpy()
    #
    #         fall_prediction_result = logits_numpy[0]
    #         nonfall_prediction_result = logits_numpy[1]
    #
    #         if fall_prediction_result > nonfall_prediction_result:
    #             print('fall')
    #         elif nonfall_prediction_result > fall_prediction_result:
    #             print('non fall c')
    #     else:
    #         print('non fall b')
    #
    # else:
    #     print('non fall a')
    ##############


    response = {
        'success': True,
        'message': 'Fall or Non fall detected',
        'error': '',
        'payload': 'results'
    }
    return response


# Get Time Diff
def get_time_diff(from_time, to_time):
    duration = to_time - from_time
    time_diff_calculated = duration.microseconds
    return time_diff_calculated


# Get x y z from the frame
def get_median_x_y_z(frame_to_extract_x_y_z):

    xs = []
    ys = []
    zs = []

    pct = frame_to_extract_x_y_z

    # Calculating all the x, y and z
    for i in range(len(pct)):
        zt = pct[i][0] * np.sin(pct[i][2]) + 0.0
        xt = pct[i][0] * np.cos(pct[i][2]) * np.sin(pct[i][1])
        yt = pct[i][0] * np.cos(pct[i][2]) * np.cos(pct[i][1])
        xs.append(xt)
        ys.append(yt)
        zs.append(zt)

    median_x_y_z = [np.median(xs), np.median(ys), np.median(zs)]
    return median_x_y_z


# Get velocity x y z
def get_velocity_x_y_z(from_xyz, to_xyz, time_diff):
    vel_x = (from_xyz[0] - to_xyz[0]) / time_diff
    vel_y = (from_xyz[1] - to_xyz[1]) / time_diff
    vel_z = (from_xyz[2] - to_xyz[2]) / time_diff
    velocity_x_y_z = [vel_x, vel_y, vel_z]
    return velocity_x_y_z


# Get acceleration x y z
def get_acceleration_x_y_z(from_velocity_xyz, to_velocity_xyz, time_diff):
    acc_x = (from_velocity_xyz[0] - to_velocity_xyz[0]) / time_diff
    acc_y = (from_velocity_xyz[1] - to_velocity_xyz[1]) / time_diff
    acc_z = (from_velocity_xyz[2] - to_velocity_xyz[2]) / time_diff
    acceleration_x_y_z = [acc_x, acc_y, acc_z]
    return acceleration_x_y_z


# Detect fall Rahul
@app.route('/detect_fall_batman_rahul', methods=['POST'])
def detect_fall():

    # 2 feet
    # sensor_height = 0.6  # meter
    # sensor_tilt_angle = 2.0  # degrees
    sensor_height = 1.8
    sensor_tilt_angle = 10.0  # degrees
    height_of_person = 0
    height_of_person_using_max_z = 0
    data = request.form

    date_time = data['date_time']
    mac_address = data['mac_address']
    stacked_v6_data = json.loads(data['stacked_v6_data'])
    stacked_v8_data = json.loads(data['stacked_v8_data'])
    stacked_date_time = json.loads(data['stacked_date_time'])
    mid_frame_index = round((len(stacked_v6_data) / 2) + 0.1) - 1

    # Converting date time string to date time
    first_frame_date_time = datetime.strptime(stacked_date_time[0], '%Y-%m-%d %H:%M:%S.%f')
    mid_frame_date_time = datetime.strptime(stacked_date_time[mid_frame_index], '%Y-%m-%d %H:%M:%S.%f')
    last_frame_date_time = datetime.strptime(stacked_date_time[-1], '%Y-%m-%d %H:%M:%S.%f')

    # Finding Time diff
    time_diff_first_and_mid_frame = get_time_diff(first_frame_date_time, mid_frame_date_time)
    time_diff_mid_and_last_frame = get_time_diff(mid_frame_date_time, last_frame_date_time)

    # Finding Median x y z
    first_frame_median_x_y_z = get_median_x_y_z(stacked_v6_data[0])
    mid_frame_median_x_y_z = get_median_x_y_z(stacked_v6_data[mid_frame_index])
    last_frame_median_x_y_z = get_median_x_y_z(stacked_v6_data[-1])

    # Velocity x y z
    velocity_first_and_mid_x_y_z = get_velocity_x_y_z(first_frame_median_x_y_z, mid_frame_median_x_y_z,
                                                      time_diff_first_and_mid_frame)
    velocity_mid_and_last_x_y_z = get_velocity_x_y_z(mid_frame_median_x_y_z, last_frame_median_x_y_z,
                                                     time_diff_mid_and_last_frame)
    acceleration_x_y_z = get_acceleration_x_y_z(velocity_first_and_mid_x_y_z, velocity_mid_and_last_x_y_z,
                                                time_diff_mid_and_last_frame)

    combined_v8_data = []

    for v8 in stacked_v8_data:
        combined_v8_data.extend(v8)

    # is_room_occupied = detect_occupancy(combined_v8_data)
    # print(len(combined_v8_data))
    is_room_occupied = len(combined_v8_data) > 0

    centroid_x = 0
    centroid_y = 0
    centroid_z = 0


    # if the room is occupied
    if is_room_occupied:
        # print('Room occupied')
        combined_frames = pd.DataFrame()

        for each_frame in stacked_v6_data:
            each_frame_x_y_z_without_noise = get_x_y_z_without_noise(each_frame)

            # Getting Height
            if centroid_x == 0:
                x_data = each_frame_x_y_z_without_noise[0].to_numpy()
                y_data = each_frame_x_y_z_without_noise[0].to_numpy()
                z_data = each_frame_x_y_z_without_noise[0].to_numpy()

                # Checking if
                if x_data.size != 0:
                    centroid_x = abs(np.sum(x_data) / x_data.size)
                    centroid_y = abs(np.sum(y_data) / y_data.size)
                    centroid_z = abs(np.sum(z_data) / z_data.size)
                    height_of_person = get_person_height(sensor_height, sensor_tilt_angle, centroid_x, centroid_y, centroid_z)
                    # print(height_of_person)
                # else:
                    # print('0 height')

            concat_frames = pd.concat([combined_frames, each_frame_x_y_z_without_noise])
            combined_frames = concat_frames

        if combined_frames.shape[0] > 0:
            height_of_person_using_max_z = abs(z_max(concat_frames.to_numpy())[2])

        # real_time_test_data = np.load('/Users/samprabin/Documents/xealei-fall/AI/datasetsNPY/fall/test/p6-F111.npy')
        real_time_test_data = combined_frames.to_numpy()

        logits = test_model(np.stack([real_time_test_data]), training=False)
        probs = tf.math.sigmoid(logits)
        max_idxs = tf.math.argmax(probs, axis=1)
        one_hot_q = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)
        results = one_hot_q[0].numpy()

        logits_numpy = logits[0].numpy()

        fall_prediction_result = logits_numpy[0]
        nonfall_prediction_result = logits_numpy[1]

        # print(height_of_person_using_max_z)
        # print(probs[0].numpy()[0])

        if fall_prediction_result > nonfall_prediction_result:
            if float(probs[0].numpy()[0]) > float(0.80):
                print(date_time, probs[0].numpy()[0], 'fall')
            else:
                print(date_time, probs[0].numpy()[0])
        elif nonfall_prediction_result > fall_prediction_result:
            print(date_time,  'non fall')
    else:
        # print('non fall')
        print(date_time, 'Room empty')

    response = {
        'success': True,
        'message': 'Fall or Non fall detected',
        'error': '',
        'payload': 'results'
    }
    return response


def test_data_sam():
    directory_to_test = 'testdataset'
    fall_records = []
    non_fall_records = []
    for filename in os.listdir(directory_to_test):
        if filename.split('.')[-1] == 'npy':
            real_time_test_data = np.load(directory_to_test + '/' + filename)

            logits = test_model(np.stack([real_time_test_data]), training=False)
            probs = tf.math.sigmoid(logits)
            max_idxs = tf.math.argmax(probs, axis=1)
            one_hot_q = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)
            results = one_hot_q[0].numpy()

            logits_numpy = logits[0].numpy()

            fall_prediction_result = logits_numpy[0]
            nonfall_prediction_result = logits_numpy[1]

            if fall_prediction_result > nonfall_prediction_result:
                fall_records.append(0)
            elif nonfall_prediction_result > fall_prediction_result:
                non_fall_records.append(1)

    print(fall_records)
    print(non_fall_records)


test_data_sam()

# starting ml api
if __name__ == '__main__':
    app.run(threaded=True)
    log = logging.getLogger('werkzeug')
    log.disabled = True