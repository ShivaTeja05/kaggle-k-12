import io
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from k12.adaptive import AdaptiveDiagnosis, class_weights, fit_head, train_cached, predict_cached
from k12.core import make_folds, fingerprint


class AdaptationContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.hidden = torch.randn(24,12)
        self.weight = torch.randn(4,12)
        self.bias = torch.randn(4)

    def test_new_layer_preserves_original_predictions_at_initialization(self):
        model = AdaptiveDiagnosis(self.weight,self.bias).eval()
        logits,error,delta = model(self.hidden)
        torch.testing.assert_close(logits,F.linear(self.hidden,self.weight,self.bias),rtol=0,atol=0)
        self.assertEqual(float(delta.detach().abs().sum()),0)
        self.assertEqual(float(error.detach().abs().sum()),0)

    def test_training_changes_adapter_but_preserves_frozen_root_weights(self):
        labels = torch.arange(len(self.hidden)) % 4
        model = fit_head(self.hidden,labels,self.weight,self.bias,0,'residual',steps=8)
        torch.testing.assert_close(model.root.weight,self.weight,rtol=0,atol=0)
        torch.testing.assert_close(model.root.bias,self.bias,rtol=0,atol=0)
        self.assertGreater(float(model.up.weight.detach().abs().sum()),0)
        self.assertFalse(model.root.weight.requires_grad)

    def test_saved_adapter_reproduces_predictions(self):
        model = fit_head(self.hidden,torch.arange(24)%4,self.weight,self.bias,0,'residual',steps=3)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        restored = AdaptiveDiagnosis(self.weight,self.bias).eval()
        restored.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(restored(self.hidden)[0],model(self.hidden)[0],rtol=0,atol=0)

    def test_rare_classes_get_bounded_weights_without_inventing_samples(self):
        weights=class_weights(torch.tensor([0]*20+[1]*2+[2]),4)
        self.assertGreater(weights[2],weights[1]);self.assertGreater(weights[1],weights[0])
        self.assertEqual(weights[3],0);self.assertLessEqual(float(weights.max()),3)

    def test_conservative_layer_limits_logit_changes_and_keeps_identity_initialization(self):
        model=AdaptiveDiagnosis(self.weight,self.bias,'conservative').eval()
        baseline=F.linear(self.hidden,self.weight,self.bias)
        torch.testing.assert_close(model(self.hidden)[0],baseline,rtol=0,atol=0)
        with torch.no_grad():model.up.weight.fill_(100)
        self.assertLessEqual(float((model(self.hidden)[0]-baseline).detach().abs().max()),.50001)
        self.assertEqual(model.rank,8)

    def test_cached_workflow_checks_partition_and_complete_test_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            data=root/'data';data.mkdir();output=root/'adapted'
            classes=['ALG-01','CAL-01','NONE']
            frame=pd.DataFrame({'trace_id':[f'r{i}' for i in range(25)],
                'question_id':[f'q{i%3}' for i in range(25)],
                'transcription_steps':[str(i) for i in range(25)],
                'root_code':[classes[i%3] for i in range(25)]})
            frame.to_csv(data/'train.csv',index=False)
            test=pd.DataFrame({'trace_id':['t1','t2','t3']});test.to_csv(data/'test.csv',index=False)
            folds=make_folds(frame);fit=frame.loc[folds!=0];val=frame.loc[folds==0]
            manifest={'classes':classes,'fold':0,'split_seed':2026,
                      'train_sha256':fingerprint(data/'train.csv'),
                      'checkpoint_trained_ids':fit.trace_id.tolist(),
                      'checkpoint_metadata_sha256':'synthetic-metadata',
                      'base_config_sha256':'synthetic-config'}
            (cache/'manifest.json').write_text(json.dumps(manifest))
            w=self.weight[:3];b=self.bias[:3]
            np.savez(cache/'base_head.npz',weight=w.numpy(),bias=b.numpy(),classes=np.asarray(classes))
            for name,part in [('fitting',fit),('validation',val),('test',test)]:
                h=torch.randn(len(part),12)
                np.savez(cache/(name+'_features.npz'),ids=np.asarray(part.trace_id,dtype=str),
                         hidden=h.numpy(),logits=F.linear(h,w,b).numpy())
            options=SimpleNamespace(features=str(cache),output=str(output),data=str(data),steps=2,seed=7)
            with contextlib.redirect_stdout(io.StringIO()):train_cached(options)
            options.checkpoint=str(output/'residual.pt');options.output=str(root/'prediction')
            with contextlib.redirect_stdout(io.StringIO()):predict_cached(options)
            predicted=pd.read_csv(root/'prediction/submission.csv',keep_default_na=False)
            self.assertEqual(set(predicted.trace_id),set(test.trace_id))
            self.assertTrue(all(set(c.split())==set(classes) for c in predicted.root_code))
            manifest['checkpoint_trained_ids'].append(val.trace_id.iloc[0])
            (cache/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'checkpoint partition'):train_cached(options)


if __name__ == '__main__':unittest.main()
