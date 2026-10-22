import unittest
import numpy as np
import torch
from packed_readback import PackedReadback


class Wrapped:
    def __init__(self,tensor):self.torch=tensor


class PackedReadbackTests(unittest.TestCase):
    def test_exact_mixed_dtypes_and_shapes(self):
        fields={'q':torch.arange(28,dtype=torch.float32).reshape(1,28),
                'force':Wrapped(torch.arange(6,dtype=torch.float32).reshape(1,1,2,3)),
                'precise':torch.tensor([1.+2.**-40],dtype=torch.float64),
                'large_integer':torch.tensor([2**60+123],dtype=torch.int64),
                'mask':torch.tensor([True,False]),'empty':torch.empty(0,dtype=torch.float32)}
        reader=PackedReadback();actual=reader.read(fields)
        for key,value in fields.items():
            expected=getattr(value,'torch',value).numpy()
            np.testing.assert_array_equal(actual[key],expected)
            self.assertEqual(actual[key].dtype,expected.dtype)
            self.assertEqual(actual[key].shape,expected.shape)
        self.assertEqual(reader.cpu_transfers,4)
        fields['q']+=100
        self.assertEqual(actual['q'][0,0],0.)

    def test_noncontiguous(self):
        tensor=torch.arange(12,dtype=torch.float32).reshape(3,4).T
        np.testing.assert_array_equal(PackedReadback().read({'view':tensor})['view'],tensor.numpy())

    def test_nonfinite_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):PackedReadback().read({'q':torch.tensor([value])})

    def test_empty_mapping_and_unsupported_cast_rejected(self):
        with self.assertRaises(ValueError):PackedReadback().read({})
        with self.assertRaises(TypeError):PackedReadback().read({'q':torch.ones(2,dtype=torch.bfloat16)})


if __name__=='__main__':unittest.main()
