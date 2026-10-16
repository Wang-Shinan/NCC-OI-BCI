from dataclasses import replace

import pytest

from bci_dayloop.serving.protocol import PredictionResult, ProtocolError, prediction_message
from bci_dayloop.serving.reve_dual_axis import combine_axis_predictions, DUAL_SEMANTICS


def pair(names, probabilities, logits):
    return PredictionResult('request', 'obs', 12, 'segment', 0, names[0], names,
                            probabilities, max(probabilities), 'revision', logits=logits)


def test_preserves_independent_pairs_and_raw_logits():
    x = pair(('left_hand', 'right_hand'), (.8, .2), (12., 10.))
    y = pair(('tongue', 'rest'), (.3, .7), (-20., -18.))
    result = combine_axis_predictions(x, y, 'dual')
    message = prediction_message(result)
    assert message['probabilities'] == [.8, .2, .3, .7]
    assert message['logits'] == [12., 10., -20., -18.]
    assert message['output_semantics'] == DUAL_SEMANTICS
    assert message['task'] == 'smr_dual_axis'
    assert result.window_id == 12


@pytest.mark.parametrize('change', [dict(request_id='other'), dict(window_id=13), dict(segment_id='other'),
    dict(class_names=('rest', 'tongue')), dict(probabilities=(.1, .3)),
    dict(probabilities=(float('nan'), .5)), dict(logits=None), dict(logits=(0., float('inf')))])
def test_rejects_mixed_windows_or_invalid_axis(change):
    x = pair(('left_hand', 'right_hand'), (.5, .5), (0., 0.))
    y = pair(('tongue', 'rest'), (.5, .5), (0., 0.))
    with pytest.raises(ProtocolError):
        combine_axis_predictions(x, replace(y, **change), 'dual')
