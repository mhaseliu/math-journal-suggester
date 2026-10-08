import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from journal_suggester.gb10_runtime import configure


class RuntimeTests(unittest.TestCase):
    def test_invalid_limits_fail_before_loading_torch(self):
        for value in ("-1", "0", "nan", "inf"):
            with self.subTest(value=value), patch.dict(os.environ, {"JOURNAL_CUDA_MEMORY_GIB": value}, clear=True):
                with self.assertRaises(ValueError):
                    configure()

    def test_memory_runtime_refuses_laptop(self):
        with patch.dict(os.environ, {"JOURNAL_CUDA_MEMORY_GIB": "32"}, clear=True), patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(RuntimeError):
                configure()

    def test_inherited_limit_is_applied_and_cannot_take_all_memory(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "NVIDIA GB10"
        cuda.get_device_properties.return_value = SimpleNamespace(total_memory=128 * 2**30)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}), patch("platform.machine", return_value="aarch64"):
            with patch.dict(os.environ, {"JOURNAL_CUDA_MEMORY_GIB": "32"}, clear=True):
                configure()
                cuda.set_per_process_memory_fraction.assert_called_once_with(0.25)
            with patch.dict(os.environ, {"JOURNAL_CUDA_MEMORY_GIB": "96"}, clear=True):
                with self.assertRaises(ValueError):
                    configure()


if __name__ == "__main__":
    unittest.main()
