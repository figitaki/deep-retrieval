import caffe
import numpy as np


class NormalizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        self.eps = 1e-8

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data / np.expand_dims(
            self.eps + np.sqrt((bottom[0].data ** 2).sum(axis=1)), axis=1)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "Backward pass not supported with this implementation")


class AggregateLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        tmp_shape[0] = 1
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data.sum(axis=0)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "Backward pass not supported with this implementation")
