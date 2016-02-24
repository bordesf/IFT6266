#Here is the function inspired by the fuel MinimumImageDimension function
#to resize all the images in the same size.
#If you want to use it, you have simply to add it in the image.py file
#in the fuel transformers folder.

class MaximumImageDimensions(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to maximum dimensions.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    maximum_shape : 2-tuple
        The maximum `(height, width)` dimensions every image must have.
        Images whose height and width are larger than these dimensions
        are passed through as-is.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.

    """
    def __init__(self, data_stream, maximum_shape, resample='nearest',
            **kwargs):
        self.maximum_shape = maximum_shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(MaximumImageDimensions, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        max_height, max_width = self.maximum_shape
        original_height, original_width = example.shape[-2:]
        if original_height > max_height or original_width > max_width:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            im = numpy.array(im.resize((max_width, max_height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example


