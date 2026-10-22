"""Small tests that run from the standalone code-repository root."""

import unittest

import polars as pl

import crystoria
import reproduce_symmetry


class PublicReleaseTests(unittest.TestCase):
    def test_crystal_system_boundaries(self):
        for number, name in ((1, "Triclinic"), (3, "Monoclinic"), (16, "Orthorhombic"),
                             (75, "Tetragonal"), (143, "Trigonal"), (168, "Hexagonal"),
                             (195, "Cubic"), (230, "Cubic")):
            self.assertEqual(reproduce_symmetry.crystal_system(number), name)
        for number in (0, 231):
            with self.assertRaises(ValueError):
                reproduce_symmetry.crystal_system(number)

    def test_full_matrix_and_denominator(self):
        frame = pl.DataFrame({"record_id": ["a", "b", "c"],
                              "source_spacegroup": [1, 1, 230],
                              "relaxed_spacegroup": [1, 2, 230]})
        matrix, systems, result = reproduce_symmetry.summarize(frame)
        self.assertEqual(matrix.shape, (230, 230))
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(result["same_spacegroup"], 2)
        self.assertEqual(result["changed_spacegroup"], 1)
        self.assertEqual(systems[0]["structures"], 2)
        self.assertIsNone(systems[1]["same_percent"])

    def test_invalid_assignments_and_duplicates_rejected(self):
        for ids, numbers in ((["a", "a"], [1, 2]), (["a", "b"], [0, 2]),
                             (["a", "b"], [None, 2])):
            frame = pl.DataFrame({"record_id": ids, "source_spacegroup": numbers,
                                  "relaxed_spacegroup": [1, 2]})
            with self.assertRaises(ValueError):
                reproduce_symmetry.summarize(frame)

    def test_portable_api_present(self):
        for name in ("verify", "statistics", "read_structure", "metric_table"):
            self.assertTrue(callable(getattr(crystoria, name)))


if __name__ == "__main__":
    unittest.main()
