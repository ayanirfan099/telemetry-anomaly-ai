import unittest
import numpy as np
from model import TelemetryModel,validate_record,ROOT,read_csv
from generate_data import FEATURES

class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=TelemetryModel()
    def test_clear_extreme_gets_flagged(self):
        record=dict(zip(FEATURES,[99,900,0.7,2000]))
        self.assertTrue(self.model.predict_batch([record])['results'][0]['is_anomaly'])
    def test_typical_record(self):
        record=dict(zip(FEATURES,[42,95,0.01,500]))
        self.assertFalse(self.model.predict_batch([record])['results'][0]['is_anomaly'])
    def test_invalid_records(self):
        for value in [float('nan'),float('inf'),True,-1,101,'42']:
            row=dict(zip(FEATURES,[value,100,0.01,500]))
            with self.assertRaises(ValueError):validate_record(row)
        with self.assertRaises(ValueError):validate_record({})
    def test_invalid_batches(self):
        for value in [[],None,{},[{}]*501]:
            with self.assertRaises(ValueError):self.model.predict_batch(value)
    def test_training_labels_not_used(self):
        self.assertEqual(self.model.model.n_features_in_,len(FEATURES))
    def test_test_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):self.model.evaluate(ROOT/'data/train.csv')
    def test_drift_separates_shifted_batch(self):
        _,values=read_csv(ROOT/'data/calibration.csv')
        normal=[dict(zip(FEATURES,row)) for row in values[:100]]
        shifted=[{**row,'latency_ms':row['latency_ms']+400} for row in normal]
        a=self.model.predict_batch(normal)['population_stability_index']['latency_ms']
        b=self.model.predict_batch(shifted)['population_stability_index']['latency_ms']
        self.assertGreater(b,a)
        self.assertIsNone(self.model.predict_batch(normal[:2])['population_stability_index'])
    def test_evaluation_reports_false_positives(self):
        report=self.model.evaluate();self.assertEqual(report['test_rows'],600)
        self.assertGreater(report['roc_auc'],0.5)
        self.assertIn('false_positive_rate',report)

if __name__=='__main__':unittest.main()
