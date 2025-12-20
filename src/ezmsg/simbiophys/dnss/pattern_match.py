import numpy as np
import scipy.signal


def find_pattern_start(
    data: np.ndarray,
    full_pattern: np.ndarray,
    pattern_window: tuple[int, int] | None = None,
    subpattern_samples: int | None = None,  # e.g., use 30_000 for Neural Signal repeating sine waves
) -> int:
    n_pattern = full_pattern.shape[1]
    if pattern_window is None:
        pattern_window = [0, full_pattern.shape[1]]
    n_window = int(np.diff(pattern_window))
    # In a worst-case scenario, our `data` might start exactly 1 sample after our pattern_window, in which case
    # we need to finish the current pattern and continue until we have reached the next pattern and the
    # pattern window.
    req_samples = n_window - 1 + n_pattern
    assert data.shape[1] >= req_samples, f"At least {req_samples} required to ensure optimal match."

    search_data = data[:, :req_samples]

    # Search for bursts in each channel independently.
    #  Note: I tried to do a multi-channel search with correlate2d but it was incredibly slow.
    #  A multi-channel search might be made faster with pytorch.
    match_onsets = np.zeros((search_data.shape[0],), dtype=int)
    for chan_ix, chan_data in enumerate(search_data):
        xcorr = scipy.signal.correlate(
            chan_data,
            full_pattern[chan_ix % full_pattern.shape[0], pattern_window[0] : pattern_window[1]],
            mode="valid",
        )
        match_onsets[chan_ix] = np.argmax(xcorr)
    match_onset = int(np.median(match_onsets[:96]))  # Simple hack for when using HDMI

    if subpattern_samples is not None:
        # The signal might have a subpattern that can cause mis-matches at subpattern repeat intervals.
        # Count how many channels matched best at +/- 1 subpattern repeat relative to the match.
        # If -1 is bigger than +1 then we may have detected window offset, not onset.
        bin_edges = [
            match_onset + shift + win_half
            for shift in [-subpattern_samples, 0, subpattern_samples]
            for win_half in [-10, 10]
        ]
        bin_counts = np.histogram(match_onsets, bin_edges)[0]
        if bin_counts[0] > bin_counts[4]:
            match_onset -= subpattern_samples

    # Do a multi-channel alignment in a small space around the putative window onset.
    test_shifts = np.arange(-5, 6)
    tiled_burst = np.tile(
        full_pattern[:, pattern_window[0] : pattern_window[1]],
        (search_data.shape[0] // full_pattern.shape[0], 1),
    )
    rms = np.zeros((test_shifts.size,))
    for ix, shift in enumerate(test_shifts):
        temp = search_data[:, match_onset + shift : match_onset + shift + n_window]
        rms[ix] = np.sqrt(np.mean((temp - tiled_burst) ** 2))
    match_onset += test_shifts[np.argmin(rms)]

    # match_onset tells us when we matched the window. We want to know when the full pattern was matched.
    return (match_onset - pattern_window[0]) % n_pattern
