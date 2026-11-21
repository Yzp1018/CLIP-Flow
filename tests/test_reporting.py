import csv
import json
from pathlib import Path

import numpy as np
import pytest

from clipflow.baselines import compare
from clipflow.figures import histograms, dimensions
from clipflow.experiments import GROUPS


def write_scores(path, prefix):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['dataset','image_id','label','score'])
        writer.writeheader()
        for i,(label,score) in enumerate([(0,-2),(0,-1),(1,1),(1,2)]):
            writer.writerow(dict(dataset='progan',image_id=prefix+str(i),label=label,score=score))


def test_external_baseline_metrics_use_held_out_calibration(tmp_path):
    test, validation = tmp_path/'test.csv',tmp_path/'val.csv'
    write_scores(test,'test');write_scores(validation,'val')
    result = compare(test,tmp_path/'results.json',validation)
    assert result['mean_ap']==1 and result['mean_accuracy']==1
    with pytest.raises(ValueError, match='overlap'):
        compare(test,tmp_path/'invalid.json',test)


def test_figures_render_from_run_artifacts(tmp_path):
    for mode in ['P','NP']:
        root=tmp_path/mode;root.mkdir()
        for name in ['progan','stargan','guided','ldm_200']:
            np.savez(root/(name+'.npz'),labels=[0,0,1,1],scores=[0,1,2,3])
    histograms(tmp_path/'P',tmp_path/'NP',tmp_path/'hist.png')
    assert (tmp_path/'hist.png').stat().st_size>1000
    for name in GROUPS['dimensions']:
        folder=tmp_path/'dims'/name;(folder/'benchmark').mkdir(parents=True)
        (folder/'training.json').write_text(json.dumps({'settings':{'dimension':128},'trainable_parameters':123520}))
        (folder/'benchmark/metrics.json').write_text(json.dumps({'mean_ap':0.9}))
    dimensions(tmp_path/'dims',tmp_path/'dimensions.png')
    assert (tmp_path/'dimensions.png').stat().st_size>1000
