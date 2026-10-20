"""Pure production-manifest admission contracts; no file or network access."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import models


def example():
    return {'schema': 1, 'pair': 'qwen3-8b-nla', 'models': [
        {'role': role, 'repo': 'owner/model', 'revision': 'a' * 40,
         'index': 'model.safetensors',
         'files': [{'path': 'model.safetensors', 'bytes': 10, 'sha256': 'b' * 64}]}
        for role in ('target', 'verbalizer')]}


class ManifestContracts(unittest.TestCase):
    def test_valid_manifest_is_preserved(self):
        value = example()
        before = copy.deepcopy(value)
        self.assertEqual(models.validate(value), before)
        self.assertEqual(value, before)

    def test_artifacts_must_be_relative_unique_paths(self):
        for path in ('../weights', '/weights', 'a/../../weights', 'a\\weights', './weights'):
            value = example()
            value['models'][0]['files'][0]['path'] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                models.validate(value)
        value = example()
        value['models'][0]['files'] *= 2
        with self.assertRaises(ValueError):
            models.validate(value)

    def test_revisions_sizes_and_digests_are_explicit(self):
        for field, invalid in (('revision', 'main'), ('bytes', -1), ('bytes', True), ('sha256', 'unknown')):
            value = example()
            owner = value['models'][0] if field == 'revision' else value['models'][0]['files'][0]
            owner[field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                models.validate(value)

    def test_roles_and_index_are_required(self):
        value = example()
        value['models'][1]['role'] = 'target'
        with self.assertRaises(ValueError):
            models.validate(value)
        value = example()
        value['models'][0]['index'] = 'absent'
        with self.assertRaises(ValueError):
            models.validate(value)


class PreparationContracts(unittest.TestCase):
    def setUp(self):
        self.data = example()
        self.files = {role + '/model.safetensors': [10, 100, 100, 1, 2]
                      for role in ('target', 'verbalizer')}
        self.receipt = models.installation(self.data, self.files)

    def test_same_installation_is_reusable(self):
        self.assertTrue(models.prepared(copy.deepcopy(self.data), self.receipt,
                                        copy.deepcopy(self.files)))
        # JSON object key order is not part of model identity.
        reordered = dict(reversed(list(self.data.items())))
        self.assertTrue(models.prepared(reordered, self.receipt, self.files))

    def test_changed_manifest_requires_setup(self):
        for field, value in [('revision', 'c' * 40), ('repo', 'other/model')]:
            data = copy.deepcopy(self.data)
            data['models'][0][field] = value
            self.assertFalse(models.prepared(data, self.receipt, self.files))
        data = copy.deepcopy(self.data)
        data['models'][0]['files'][0]['sha256'] = 'd' * 64
        self.assertFalse(models.prepared(data, self.receipt, self.files))

    def test_missing_replaced_or_modified_files_require_setup(self):
        for index in range(5):
            files = copy.deepcopy(self.files)
            files['target/model.safetensors'][index] += 1
            self.assertFalse(models.prepared(self.data, self.receipt, files))
        self.assertFalse(models.prepared(self.data, self.receipt, {}))

    def test_missing_or_incompatible_receipt_requires_setup(self):
        for receipt in (None, {}, [], {**self.receipt, 'schema': 2}):
            self.assertFalse(models.prepared(self.data, receipt, self.files))


if __name__ == '__main__':
    unittest.main()
