import warnings

from models_oddity.consts_oddity import Consts

import torch

B, C, N, H, W = 0, 1, 2, 3, 4

class I3D(torch.nn.Module):
    """
    Inception 3D convolutional network, specifically for use with temporal
    depth of 16. Originally proposed in "Quo Vadis, Action Recognition? A
    New Model and the Kinetics Dataset" by Joao Carreira & Andrew Zisserman
    (https://arxiv.org/abs/1705.07750).

    Code inspired by previous work:
        - https://github.com/deepmind/kinetics-i3d
        - https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        - https://github.com/piergiaj/pytorch-i3d
        - https://github.com/hassony2/kinetics_i3d_pytorch

    Originally, we used the `piergiaj` and `hassony2` implementations, but
    they had some lurking bugs and inconvenciences. Furthermore, the way they
    dealt with padding was too complicated for our use case and prone to
    errors. This implementation of I3D is specifically designed to be used
    on input with shape (B, C, 16, 224, 224), where the batch size and number
    of channels can vary. Also, it does not do any runtime padding trickery,
    so it should be a bit faster and reliable.
    """

    def __init__(self,
                 num_classes,
                 modality,
                 dropout=0.0,
                 use_activation=False):

        super().__init__()

        self.name = f'backbone_i3d_{modality}'
        self.num_classes = num_classes
        self.modality = modality

        if modality == 'rgb':
            self._expect_num_channels = 3
        elif modality == 'diff':
            self._expect_num_channels = 2
        else:
            raise ValueError(f'Unknown modality: {modality}.')

        in_channels = self._expect_num_channels

        self.layer_1a = Unit(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=Padding.odd(2, 3, 2, 3, 2, 3))

        self.layer_2a = MaxPoolUnit(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=Padding.odd(0, 1, 0, 1, 0, 0))

        self.layer_2b = Unit(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1))

        self.layer_2c = Unit(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=Padding.even(1, 1, 1))

        self.layer_3a = MaxPoolUnit(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=Padding.odd(0, 1, 0, 1, 0, 0))

        self.layer_3b = Block(
            in_channels=192,
            out_channels=(64, (96, 128), (16, 32), 32))

        self.layer_3c = Block(
            in_channels=256,
            out_channels=(128, (128, 192), (32, 96), 64))

        self.layer_4a = MaxPoolUnit(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=Padding.odd(0, 1, 0, 1, 0, 1))

        self.layer_4b = Block(
            in_channels=480,
            out_channels=(192, (96, 208), (16, 48), 64))

        self.layer_4c = Block(
            in_channels=512,
            out_channels=(160, (112, 224), (24, 64), 64))

        self.layer_4d = Block(
            in_channels=512,
            out_channels=(128, (128, 256), (24, 64), 64))

        self.layer_4e = Block(
            in_channels=512,
            out_channels=(112, (144, 288), (32, 64), 64))

        self.layer_4f = Block(
            in_channels=528,
            out_channels=(256, (160, 320), (32, 128), 128))

        self.layer_5a = MaxPoolUnit(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=Padding.even(0, 0, 0))

        self.layer_5b = Block(
            in_channels=832,
            out_channels=(256, (160, 320), (32, 128), 128))

        self.layer_5c = Block(
            in_channels=832,
            out_channels=(384, (192, 384), (48, 128), 128))

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))

        self.dropout = torch.nn.Dropout(dropout)

        self.classify = Unit(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            use_bias=True,
            use_normalization=False,
            use_activation=False)

        if use_activation:
            self.activate = torch.nn.Softmax(1)
        else:
            self.activate = None

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classify(x)
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        x = torch.mean(x, 2)
        if self.activate is not None:
            x = self.activate(x)
        return x

    def extract_features(self, x):
        if not torch.jit.is_scripting():
            assert(x.shape[N] == Consts.TIME_BATCH_SIZE)
            assert(x.shape[C] == self._expect_num_channels)
            assert(x.shape[H:] == Consts.IMAGE_DIMS)
        x = self.layer_1a(x)
        x = self.layer_2a(x)
        x = self.layer_2b(x)
        x = self.layer_2c(x)
        x = self.layer_3a(x)
        x = self.layer_3b(x)
        x = self.layer_3c(x)
        x = self.layer_4a(x)
        x = self.layer_4b(x)
        x = self.layer_4c(x)
        x = self.layer_4d(x)
        x = self.layer_4e(x)
        x = self.layer_4f(x)
        x = self.layer_5a(x)
        x = self.layer_5b(x)
        x = self.layer_5c(x)
        x = self.avg_pool(x)
        return x

    def load_weights(self, weights_file):
        with TrainMode(model=self, train=False):
            state_dict_dirty = torch.load(weights_file)
            state_dict_fixed = self._fix_state_dict_classification_weights(state_dict_dirty)
            self.load_state_dict(state_dict_fixed)

    def _fix_state_dict_classification_weights(self, state_dict):
        # Collect all keys in the state dictionary that have
        # `classify` in them. Those are the state parameters
        # for the last layer that we might need to replace.
        fix_keys = [k for k in state_dict.keys() if 'classify' in k]

        # This is not expected at this point.
        if len(fix_keys) <= 0:
            raise ValueError('Cannot fix state dict that does not have '
                             'classification parameters at all.')

        # Fix all the classification module parameters by
        # replacing the weights such that they have the
        # correct shape, with values all zeros.
        for state_dict_key in fix_keys:
            old_params = state_dict[state_dict_key]
            old_device = old_params.device
            new_shape = (self.num_classes, *(old_params.size()[1:]))
            state_dict[state_dict_key] = \
                torch.zeros(size=new_shape, device=old_device)

        return state_dict

class Block(torch.nn.Module):
    """
    I3D consists of a number of similar blocks that have four
    branches that are concatenated together. This class helps
    constructing these blocks easily.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_0_out_channels, \
        (branch_1a_out_channels, branch_1b_out_channels), \
        (branch_2a_out_channels, branch_2b_out_channels), \
        branch_3b_out_channels = \
            out_channels

        self.branch_0 = \
            Unit(in_channels=in_channels,
                 out_channels=branch_0_out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1))

        self.branch_1 = torch.nn.Sequential(
            Unit(in_channels=in_channels,
                 out_channels=branch_1a_out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1)),
            Unit(in_channels=branch_1a_out_channels,
                 out_channels=branch_1b_out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=Padding.even(1, 1, 1)))

        self.branch_2 = torch.nn.Sequential(
            Unit(in_channels=in_channels,
                 out_channels=branch_2a_out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1)),
            Unit(in_channels=branch_2a_out_channels,
                 out_channels=branch_2b_out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=Padding.even(1, 1, 1)))

        self.branch_3 = torch.nn.Sequential(
            MaxPoolUnit(kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=Padding.even(1, 1, 1)),
            Unit(in_channels=in_channels,
                 out_channels=branch_3b_out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1)))

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), 1)

class Unit(torch.nn.Module):
    """
    Most layers in I3D consist of a convolutional layer, padded
    according to TensorFlow's SAME strategy, normalization and
    activation. This class allows for easily constructing such
    a unit. Note that bias, activation and normalization are not
    should be disabled at some places.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 use_bias=False,
                 use_normalization=True,
                 use_activation=True):

        super().__init__()

        self.pad = None

        padding_param = {}

        if padding is not None:
            if padding.is_odd:
                self.pad = padding.as_module()
            if padding.is_even:
                padding_param = dict(padding=padding.as_param())

        self.conv3d = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=use_bias,
            **padding_param)

        if use_normalization:
            """
            Batch normalization layers can heavily influence training, even more so when doing transfer
            learning. I'm currently working on replacing BN altogether. Until then, please note the fo-
            llowing few things. The Keras implementation does not have scaling turned on for BN. In the
            docs it says "When the next layer is linear (e.g. nn.relu), this can be disabled since the 
            scaling will ben done by the next layer.". It is turned off here as well.

            Furthermore, there are bunch of "modes" in which BN can operate:

                1. A normal BN layer in training model does the following:
                    - Affine layers (weight and bias) are governed by backprop.
                    - Mean and variance are based on batch statistics, so they are quite volatile.
                2. A BN layer in inference mode (sometimes called "frozen" when used during training):
                    - Affine layers are still updated by backprop (like 1).
                    - Mean and variance are kept as running statistics, and span batches.
                3. A BN layer that is completely frozen:
                    - Does not update affine parameters (alpha & beta).
                    - Uses inference mode for statistics (like 2).

            In PyTorch, each of the states can be programmed as such:

                1. Have BN parameters enlisted with optimizer. Make sure `requires_grad==True` for all 
                   weights. Make sure to have the layer be set with `train=True` to make sure batch st-
                   ats are used.

                2. Set eval-mode on BN layers (i.e. `train=False`). Parameters should be enlisted with
                   autograd.

                3. Set eval-mode on BN layers, and set `requires_grad` to `False` (to prevent affine w-
                   eights tensors from being updated.

            One can also force use of batch stats during evaluation by setting `track_running_states=F-
            alse` on BN layers. Not sure why anyone would want this but it is possible.

            The behavior of BN in this implementation might have changed compared to the original Keras
            one. There, the model was in state 1 during base model training and a very weird state dur-
            ing extension training: The affine weights were frozen but the model was still using batch
            statistics rather than inference stats. The current behavior is better in theory, but bewa-
            re of the difference. Using the old implementation in TF2 now would yield comparable resul-
            ts to what we are doing here because the semantics of `trainable` were changed.

            NOTE: Even when the BN layer is in mode 3 in PyTorch, the weight and bias can theoretically
            still be adjusted one epoch longer than expected if an optimizer with any kind of running 
            average in it (basically almost everything except basic SGD) is used. This is the case if
            `requires_grad` is put to False after a while. The idea behind this is that the gradients
            are still present and thus still have influence. `weight.grad = None` and/or `bias.grad =
            None` make sure this does not happen. Or a better fix (already in place): do not give these
            variables to the optimizer at all.

            Also, in the way we're saving and loading models (by saving/loading state dicts), grads are
            not saved. So, this is not a problem when training a base model, saving it and later reusi-
            ng it for extension training (when using mode 3 from the start of extension training).

            NOTE: Even if this is done, `running_var` is still calculated and applied in the normaliza-
            tion. If needed to disable this completely, keep `running_var` fixed at 1.0,1.0,... Keep in
            mind that this only works in inference mode; if you train the layer, the `running_var` will
            be updated.
            """
            self.norm = torch.nn.BatchNorm3d(
                num_features=out_channels,
                eps=0.001, momentum=0.01)
            self.norm.weight.requires_grad_(False)
        else:
            self.norm = None

        if use_activation:
            self.activate = torch.nn.ReLU()
        else:
            self.activate = None

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv3d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

class MaxPoolUnit(torch.nn.Module):
    """
    At various points within I3D, there are max pooling units.
    They appear inside blocks, but there are also some separate
    uses of max pooling. Max pooling is wrapped in here so that
    it uses padding that is similar to TensorFlow's.
    """

    def __init__(self,
                 kernel_size,
                 stride,
                 padding=None):

        super().__init__()

        if padding is not None:
            self.pad = padding.as_module(allow_expand=True)

        self.maxpool3d = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        # Newer PyTorch version emit a warning for an upcoming change in
        # the order of parameter that does not affect us. It does contaminate
        # the logs though so we're ignoring it here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            x = self.maxpool3d(x)
        return x

class Padding:
    """
    Because PyTorch does not have the SAME padding strategy that
    TensorFlow has, we need to do some additional handling of
    padding throughout the model. This data class helps with that.
    """

    def __init__(self, kind, values):
        if kind not in ['even', 'odd']:
            raise ValueError(f'Unknown padding kind: {kind}.')

        if kind == 'even' and len(values) != 3:
            raise ValueError(f'Even padding must have 3 dimensions, but received {len(values)}.')

        if kind == 'odd' and len(values) != 6:
            raise ValueError(f'Odd padding must have 6 dimensions, but received {len(values)}.')

        self.kind = kind
        self.values = values

    @staticmethod
    def even(*values):
        return Padding('even', values)

    @staticmethod
    def odd(*values):
        return Padding('odd', values)

    @property
    def is_even(self):
        return self.kind == 'even'

    @property
    def is_odd(self):
        return self.kind == 'odd'

    def as_param(self):
        if self.kind == 'even':
            return self.values
        else:
            raise ValueError('Only even padding can be used as parameter.')

    def as_module(self, allow_expand=False):
        if self.kind == 'odd':
            return torch.nn.ConstantPad3d(tuple(self.values), 0.)
        else:
            if self.kind == 'even' and allow_expand:
                values_expanded = (
                    self.values[0], self.values[0],
                    self.values[1], self.values[1],
                    self.values[2], self.values[2])

                return torch.nn.ConstantPad3d(values_expanded, 0.)
            else:
                raise ValueError('Only even and odd padding can be used as parameter.')

class TrainMode():
    """
    This context manager is used to transiently put a model
    in train or eval mode for the duration of the context.
    """

    def __init__(self, model, train):
        self.model = model
        self.model_prev_train = None
        self.train = train

    def __enter__(self):
        self.model_prev_train = self.model.training
        self.model.train(self.train)

    def __exit__(self, _type, _value, _traceback):
        self.model.train(self.model_prev_train)
