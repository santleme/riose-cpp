#ifndef CAMPO_CORE_H
#define CAMPO_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v1. Scalar inputs/outputs use the units in their names. Returns 0 on
   success and -1 for invalid inputs. Output pointers may be NULL. */
uint32_t campo_core_abi_version(void);
int32_t campo_step_patch_v1(
    double herbage_kg_dm_ha,
    double soil_water_mm,
    double soil_capacity_mm,
    double precipitation_mm,
    double irrigation_mm,
    double potential_et_mm,
    double net_growth_kg_dm_ha,
    double requested_removal_kg_dm_ha,
    double *out_herbage_kg_dm_ha,
    double *out_soil_water_mm,
    double *out_runoff_mm,
    double *out_actual_et_mm,
    double *out_unmet_removal_kg_dm_ha);

/* BR-CORTE 2023, chapter 14 tropical beef-cattle water-intake estimate.
   Outputs L/day (assuming ~1 kg/L) and 1 if all predictors are in the
   published independent-validation ranges, else 0 (extrapolation). */
int32_t campo_estimate_water_intake_brcorte_2023_v1(
    double body_weight_kg,
    double dmi_kg_dm_day,
    double tmax_celsius,
    double relative_humidity_percent,
    double *out_liters_per_day,
    uint32_t *out_within_validation);

#ifdef __cplusplus
}
#endif
#endif
