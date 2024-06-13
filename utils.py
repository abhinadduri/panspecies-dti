import featurizers

def get_featurizer(featurizer_string, *args, **kwargs):

    return getattr(featurizers, featurizer_string)(*args, **kwargs)
