import unittest

from audit_nellore_shade import br_corte_water_intake_l_day


class BrCorteWaterEquationTests(unittest.TestCase):
    def test_reproduces_published_br_corte_example(self):
        predicted = br_corte_water_intake_l_day(
            body_weight_kg=520,
            max_temperature_c=29,
            relative_humidity_percent=79,
            dry_matter_intake_kg_day=10.62,
        )
        self.assertAlmostEqual(predicted, 22.73, places=2)

    def test_rejects_relative_humidity_outside_percent_range(self):
        with self.assertRaisesRegex(ValueError, "expressed in percent"):
            br_corte_water_intake_l_day(520, 29, 101, 10.62)


if __name__ == "__main__":
    unittest.main()
