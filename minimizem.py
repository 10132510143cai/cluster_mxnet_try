import mxnet as mx

prefix = 'mymodel'
iteration = 100

# load model back
model_loaded = mx.model.FeedForward.load(prefix, iteration)