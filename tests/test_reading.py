import unittest
from types import SimpleNamespace

import torch
import numpy as np

from k12.reading_finetune import append_teacher,teacher_response,shard_order


class ReadingSupervisionContracts(unittest.TestCase):
    def test_teacher_tokens_follow_the_unchanged_classification_prefix(self):
        pixels=torch.randn(2,3)
        original={'input_ids':torch.tensor([[10,11,12]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
                  'mm_token_type_ids':torch.tensor([[0,1,0]]),'pixel_values':pixels}
        joined,length=append_teacher(original,torch.tensor([[70,71]]))
        self.assertEqual(length-1,2)
        torch.testing.assert_close(joined['input_ids'][:,:length],original['input_ids'])
        torch.testing.assert_close(original['input_ids'],torch.tensor([[10,11,12]]))
        self.assertIs(joined['pixel_values'],pixels)
        self.assertEqual(joined['attention_mask'].shape,joined['input_ids'].shape)
        torch.testing.assert_close(joined['mm_token_type_ids'],torch.tensor([[0,1,0,0,0]]))

    def test_reading_and_root_cause_are_supervision_targets(self):
        row=SimpleNamespace(root_code='CAL-01',transcription_steps='GOLD_WORK',
                            misconceptions='[{"root":true,"text":"ROOT_REASON"},{"root":false,"text":"DOWNSTREAM"}]')
        reading=teacher_response(row,0);cause=teacher_response(row,1)
        self.assertIn('GOLD_WORK',reading);self.assertNotIn('ROOT_REASON',reading)
        self.assertIn('ROOT_REASON',cause);self.assertNotIn('DOWNSTREAM',cause)
        self.assertTrue(reading.startswith('ROOT: CAL-01'))

    def test_two_gpu_shards_cover_all_fitting_rows_with_one_explicit_padding_row(self):
        order=np.arange(1199)
        left=shard_order(order,0,2);right=shard_order(order,1,2)
        self.assertEqual(len(left),600);self.assertEqual(len(right),600)
        self.assertEqual(set(left)|set(right),set(order))
        self.assertEqual(len(set(left)&set(right)),1)


if __name__=='__main__':unittest.main()
