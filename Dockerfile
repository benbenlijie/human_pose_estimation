FROM tensorflow/tensorflow:1.4.0-rc1-devel-gpu-py3

# add python library
RUN pip --no-cache-dir install \
        keras \
        pandas \
        Pillow \
        h5py && \
	apt-get update && \
	apt-get install -y vim && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	mv /usr/bin/python /usr/bin/python2 && \
	ln -s /usr/bin/python3.5 /usr/bin/python
	

WORKDIR POSE

COPY model_res/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5

COPY util util/

COPY openpose openpose/

COPY openpose/train.py train.py