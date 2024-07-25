
import unittest
import numpy as np
from aiecrs import CustomerRetentionSystem

class TestAIECRS(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 10)
        self.target = np.random.randint(2, size=100)
        self.aiecrs = CustomerRetentionSystem()

    def test_train_model(self):
        self.aiecrs.train_model(self.data, self.target)
        self.assertIsNotNone(self.aiecrs.model)

    def test_predict_churn(self):
        self.aiecrs.train_model(self.data, self.target)
        predictions = self.aiecrs.predict_churn(self.data[:5])
        self.assertEqual(len(predictions), 5)

if __name__ == '__main__':
    unittest.main()
