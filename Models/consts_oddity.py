class Consts:
    
    """
    The common part of the input shape is the same for spatial and temporal streams
    and consists of the NHW part (excluding batch size and channels).
    """
    INPUT_SHAPE_COMMON = (16, 224, 224)

    """
    Three channels for spatial input: red, green and blue.
    """
    SPATIAL_STREAM_NUM_CHANNELS = 3

    """
    Two channels for temporal input: the frame difference frame, and the acceleration
    frame. The frame difference is computed as `Di = abs(Fi+1 - Fi)` (approx.). The
    acceleration is the second derivative i.e. `Ai = abs(Di+1 - Di)` (approx.).
    """
    TEMPORAL_STREAM_NUM_CHANNELS = 2

    """
    These are the input shapes for each of the streams in the base model, excluding
    explicit batch size. We provide both CNHW (channels-first) and NHWC (channels-last)
    formats.
    """
    INPUT_SHAPE_SPATIAL_STREAM_CNHW = (SPATIAL_STREAM_NUM_CHANNELS, *INPUT_SHAPE_COMMON)
    INPUT_SHAPE_TEMPORAL_STREAM_CNHW = (TEMPORAL_STREAM_NUM_CHANNELS, *INPUT_SHAPE_COMMON)
    INPUT_SHAPE_SPATIAL_STREAM_NHWC = (*INPUT_SHAPE_COMMON, SPATIAL_STREAM_NUM_CHANNELS)
    INPUT_SHAPE_TEMPORAL_STREAM_NHWC = (*INPUT_SHAPE_COMMON, TEMPORAL_STREAM_NUM_CHANNELS)

    """
    By setting the batch size to be explicitly `None`, the deep learning backend can reason
    about it correclty (at least, in the case of TensorFlow).
    """
    INPUT_SHAPE_SPACE_BATCH_SIZE = None

    """
    The number of frames per stack (extracted from common) as expected by model.
    """
    TIME_BATCH_SIZE = INPUT_SHAPE_COMMON[0]

    """
    Frame size as expected by model.
    """
    IMAGE_DIMS = INPUT_SHAPE_COMMON[1:3]
