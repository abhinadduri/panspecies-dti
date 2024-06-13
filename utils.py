
def get_featurizer(featurizer_string, *args, **kwargs):
    from . import featurizers

    return getattr(featurizers, featurizer_string)(*args, **kwargs)
