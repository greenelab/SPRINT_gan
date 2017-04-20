from keras import backend as K
import numpy as np


def PxG(sequential, x, y, sample_weight=None, class_weight=None):
    x, y, sample_weights = sequential.model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        check_batch_axis=True)
    if sequential.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [1.]
    else:
        ins = x + y + sample_weights

    PxG_function = make_PxG_function(sequential.model)
    grad = PxG_function(ins)
    return grad

def make_PxG_function(model):
    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

    grad = model.optimizer.get_gradients(
                model.total_loss,
                model._collected_trainable_weights)

    # returns loss and metrics. Updates weights at each call.
    return K.function(inputs,
                      grad,
                      **model._function_kwargs)

# Custom PxG train on batch function
# Clip gradient per example, add noise per lot
def PxG_train_on_batch(model, x, y,
                       sample_weight=None, class_weight=None):
    """Runs a single gradient update on a single batch of data.
       From Keras
        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
            class_weight: optional dictionary mapping
                lass indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to "pay more attention" to
                samples from an under-represented class.

        # Returns
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        check_batch_axis=True)
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [1.]
    else:
        ins = x + y + sample_weights
    DP_make_train_function(model)
    outputs = model.train_function(ins)

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def DP_make_train_function(model):
    if not hasattr(model, 'train_function'):
        raise RuntimeError('You must compile your model before using it.')

    if model.train_function is None:
        inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]

        training_updates = model.optimizer.get_updates(
            model._collected_trainable_weights,
            model.constraints,
            model.total_loss)
        updates = model.updates + training_updates

        # returns loss and metrics. Updates weights at each call.
        model.train_function = K.function(inputs,
                                          [model.total_loss] + model.metrics_tensors,
                                          updates=updates,
                                          **model._function_kwargs)
