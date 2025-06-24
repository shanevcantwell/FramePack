bucket_options = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    # The 'resolution' parameter is a key to select a group of pre-defined aspect ratio
    # buckets that the model was trained on. It's not the target output resolution.
    buckets = bucket_options.get(resolution)

    # Safeguard against incorrect resolution keys or empty bucket lists.
    if not buckets:
        raise ValueError(f"No buckets are defined for the specified resolution key: {resolution}")

    min_metric = float('inf')
    best_bucket = None

    # The metric calculates which bucket's aspect ratio is closest to the input's.
    for (bucket_h, bucket_w) in buckets:
        metric = abs(h * bucket_w - w * bucket_h)
        # Using '<' instead of '<=' makes the choice deterministic if multiple buckets
        # have the same metric. It will select the first one it encounters.
        if metric < min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)

    if best_bucket is None:
        # This should be unreachable if the bucket list is not empty, but it's a good safeguard.
        raise RuntimeError(f"Could not find a suitable bucket for resolution {resolution}. This is unexpected.")

    return best_bucket
