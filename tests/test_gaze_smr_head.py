import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fit_gaze_smr_head import extract_session, evaluate
from bci_dayloop.serving.reve import REVE_TASKS, default_serve_strategy


def write_session(folder, mode='train', duration=6):
    (folder/'session.json').write_text(json.dumps(dict(status='complete', subjectId='TEST', clock=dict(sampleIndex=1900))))
    (folder/'eeg.json').write_text(json.dumps(dict(sampleRate=100, channelNames=['Fp1'])))
    np.arange(900, dtype='<f4').tofile(folder/'eeg.bin')
    events = [dict(type='session_start', data=dict(mode=mode, protocol='gaze-assisted-smr-v1', targetSec=duration)),
              dict(type='gaze_trial', data=dict(index=0, run=1, target='left', valid=True, elapsedSec=duration), eeg=dict(sampleIndex=1900))]
    (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))


def test_six_second_trial_yields_three_target_only_windows(tmp_path):
    write_session(tmp_path)
    windows, protocol = extract_session(tmp_path, 'TEST')
    assert protocol == 'gaze-assisted-smr-v1'
    assert len(windows) == 3
    assert sorted(int(w.data[0,0]) for w in windows) == [300,500,700]
    assert {w.trial_index for w in windows} == {0}
    assert {w.run for w in windows} == {1}


def test_feedback_and_wrong_subject_never_enter_training(tmp_path):
    write_session(tmp_path, mode='feedback')
    with pytest.raises(ValueError): extract_session(tmp_path, 'TEST')
    write_session(tmp_path)
    with pytest.raises(ValueError): extract_session(tmp_path, 'OTHER')


def test_evaluation_counts_trials_not_correlated_windows(tmp_path):
    write_session(tmp_path)
    windows, _ = extract_session(tmp_path, 'TEST')
    result = evaluate(np.array([[.8,.2,0,0],[.7,.3,0,0],[.1,.9,0,0]]),np.zeros(3,dtype=int),windows)
    assert result['n'] == 1
    assert result['confusion'][0][0] == 1
    assert result['balancedAccuracy'] is None


def test_gaze_task_is_separate_and_frozen():
    assert REVE_TASKS['gaze_smr'].class_names == ('left','right','up','down')
    assert REVE_TASKS['gaze_smr'].class_names != REVE_TASKS['smr_control'].class_names
    assert default_serve_strategy('gaze_smr') == 'none'


def test_binary_restricted_head_keeps_four_output_contract():
    import torch
    from fit_smr_control_head import train_linear_head
    from types import SimpleNamespace
    torch.manual_seed(4)
    head=torch.nn.Linear(2,4)
    x=torch.tensor([[-1.,0.],[-2.,0.],[1.,0.],[2.,0.]])
    y=torch.tensor([0,0,1,1])
    train_linear_head(head,x,y,epochs=50,lr=.05,class_indices=[0,1])
    assert head(x).shape == (4,4)
    probs=head(x)[:,:2].softmax(1).detach().numpy()
    windows=[SimpleNamespace(trial_index=i) for i in range(4)]
    result=evaluate(probs,y.numpy(),windows)
    assert result['balancedAccuracy']==1
    assert len(result['confusion'])==2


def test_joint_evaluation_does_not_merge_matching_trial_numbers():
    from types import SimpleNamespace
    windows=[SimpleNamespace(session_id='a',trial_index=0),SimpleNamespace(session_id='b',trial_index=0)]
    result=evaluate(np.array([[.9,.1],[.1,.9]]),np.array([0,1]),windows)
    assert result['n']==2
    assert result['balancedAccuracy']==1


def test_cued_control_requires_opt_in_and_keeps_trials_together(tmp_path):
    names = ['left', 'right', 'up', 'down'] * 2
    plan = [dict(index=i,run=1,target=t) for i,t in enumerate(names)]
    (tmp_path/'session.json').write_text(json.dumps(dict(status='complete',subjectId='TEST',clock=dict(sampleIndex=7200))))
    (tmp_path/'eeg.json').write_text(json.dumps(dict(sampleRate=100,channelNames=['Fp1'])))
    np.arange(7200,dtype='<f4').tofile(tmp_path/'eeg.bin')
    events = [dict(type='session_start',data=dict(mode='control',protocol='gaze-assisted-smr-control-v1',
        targetLayout='single-edge-v1',labelsAre='cued_edge_control_with_feedback',trials=plan))]
    for item in plan:
        start = item['index']*900+300
        events += [dict(type='gaze_phase',perf_ms=start*10,data={**item,'phase':'control'},eeg=dict(sampleIndex=start)),
                   dict(type='gaze_edge_result',perf_ms=(start+600)*10,data={**item,'elapsedSec':6,'signalIssue':None},eeg=dict(sampleIndex=start+600))]
    events.append(dict(type='session_end'))
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError): extract_session(tmp_path,'TEST')
    windows, _ = extract_session(tmp_path,'TEST',True)
    assert len(windows) == 24
    assert {w.trial_index for w in windows if w.run==3} == {4,5,6,7}
    assert all(w.start_sample >= w.trial_index*900+300 for w in windows)


def test_closed_loop_training_preserves_planned_heldout_runs(tmp_path):
    test_cued_control_requires_opt_in_and_keeps_trials_together(tmp_path)
    path=tmp_path/'events.jsonl'
    events=[json.loads(s) for s in path.read_text().splitlines()]
    cfg=events[0]['data']
    cfg.update(mode='train',protocol='gaze-assisted-smr-closed-loop-v1',predictionVisible=True)
    for trial in cfg['trials']: trial['run']=3 if trial['index']>=4 else 1
    # Stop extraction at a hit: only the pre-hit 2-second window is eligible.
    events.insert(-1,dict(type='gaze_edge_hit',data=dict(index=0),eeg=dict(sampleIndex=550)))
    path.write_text('\n'.join(json.dumps(e) for e in events))
    windows,_=extract_session(tmp_path,'TEST')
    assert {w.trial_index for w in windows if w.run==3}=={4,5,6,7}
    assert len([w for w in windows if w.trial_index==0])==1
