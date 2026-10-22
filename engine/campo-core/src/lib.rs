//! Small deterministic scientific kernel for Campo Vivo.
//!
//! This module accounts for water and herbage using externally supplied daily
//! fluxes. It does not predict plant growth, animal intake, physiology, or
//! behavior. Keeping those quantities as explicit forcings prevents an
//! accounting kernel from being mistaken for a calibrated biological model.

#![forbid(unsafe_op_in_unsafe_fn)]

use core::fmt;

macro_rules! quantity {
    ($name:ident, $unit:literal) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(f64);

        impl $name {
            pub fn new(value: f64) -> Result<Self, InputError> {
                if value.is_finite() && value >= 0.0 {
                    Ok(Self(value))
                } else {
                    Err(InputError::InvalidQuantity {
                        name: stringify!($name),
                        value,
                    })
                }
            }

            pub const fn get(self) -> f64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $unit)
            }
        }
    };
}

quantity!(ForageKgDmPerHa, "kg DM/ha");
quantity!(WaterMm, "mm");
quantity!(BodyWeightKg, "kg live weight");

pub const OBSERVATION_SCHEMA_V1: u16 = 1;
pub const ACTION_SCHEMA_V1: u16 = 1;
pub const MEMORY_BOUT_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnimalId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SiteId(pub u64);

/// Minimal individual record. Weight is an observed/calibrated state input;
/// this crate does not compute weight gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CowStateV1 {
    pub animal_id: AnimalId,
    pub body_weight: BodyWeightKg,
}

/// Canonical behavior codes are a transport ontology only. Dataset mappings
/// and ethogram definitions must be reviewed before training against them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum BehaviorKindV1 {
    #[default]
    Unknown = 0,
    Grazing = 1,
    Drinking = 2,
    Ruminating = 3,
    Resting = 4,
    Walking = 5,
    SocialContact = 6,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoutRecordV1 {
    pub behavior: BehaviorKindV1,
    /// Minutes since the start of the simulation clock.
    pub start_minute: u64,
    pub duration_minutes: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CurrentBoutV1 {
    pub behavior: BehaviorKindV1,
    /// Minutes since the start of the simulation clock.
    pub start_minute: u64,
    pub elapsed_minutes: u32,
}

/// Fixed-size, oldest-first history of completed bouts. `len` says how many
/// initial array slots are valid; unused slots must be ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentMemoryV1 {
    pub recent_bouts: [BoutRecordV1; MEMORY_BOUT_CAPACITY],
    pub len: u8,
}

impl Default for AgentMemoryV1 {
    fn default() -> Self {
        Self {
            recent_bouts: [BoutRecordV1::default(); MEMORY_BOUT_CAPACITY],
            len: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationV1 {
    pub schema_version: u16,
    pub animal: CowStateV1,
    /// Minutes since the start of the simulation clock.
    pub observed_at_minute: u64,
    pub current_bout: CurrentBoutV1,
    pub memory: AgentMemoryV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionKindV1 {
    ContinueCurrentBout = 0,
    StartGrazing = 1,
    StartDrinking = 2,
    StartRuminating = 3,
    StartResting = 4,
    MoveToSite = 5,
    SocialContact = 6,
}

/// Declarative output contract for a future learned policy. It is not itself a
/// policy and carries no claim that a chosen action is biologically optimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionV1 {
    pub schema_version: u16,
    pub kind: ActionKindV1,
    pub target_site: Option<SiteId>,
    pub requested_duration_minutes: u32,
}

impl ObservationV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != OBSERVATION_SCHEMA_V1 {
            return Err(ContractError::UnsupportedObservationVersion(
                self.schema_version,
            ));
        }
        if self.memory.len as usize > MEMORY_BOUT_CAPACITY {
            return Err(ContractError::MemoryLengthOutOfRange(self.memory.len));
        }
        let current_end = self
            .current_bout
            .start_minute
            .checked_add(self.current_bout.elapsed_minutes as u64)
            .ok_or(ContractError::InvalidBoutTime)?;
        if current_end != self.observed_at_minute {
            return Err(ContractError::InvalidBoutTime);
        }

        let mut previous_end = 0;
        for record in &self.memory.recent_bouts[..self.memory.len as usize] {
            let end = record
                .start_minute
                .checked_add(record.duration_minutes as u64)
                .ok_or(ContractError::InvalidBoutTime)?;
            if record.duration_minutes == 0
                || record.start_minute < previous_end
                || end > self.current_bout.start_minute
            {
                return Err(ContractError::InvalidBoutTime);
            }
            previous_end = end;
        }
        Ok(())
    }
}

impl ActionV1 {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != ACTION_SCHEMA_V1 {
            return Err(ContractError::UnsupportedActionVersion(self.schema_version));
        }
        if self.requested_duration_minutes == 0 {
            return Err(ContractError::ZeroActionDuration);
        }
        if (self.kind == ActionKindV1::MoveToSite) != self.target_site.is_some() {
            return Err(ContractError::InvalidActionTarget);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractError {
    UnsupportedObservationVersion(u16),
    UnsupportedActionVersion(u16),
    MemoryLengthOutOfRange(u8),
    InvalidBoutTime,
    ZeroActionDuration,
    InvalidActionTarget,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchState {
    pub herbage: ForageKgDmPerHa,
    pub soil_water: WaterMm,
    pub soil_water_capacity: WaterMm,
}

impl PatchState {
    pub fn new(
        herbage: ForageKgDmPerHa,
        soil_water: WaterMm,
        soil_water_capacity: WaterMm,
    ) -> Result<Self, InputError> {
        if soil_water > soil_water_capacity {
            return Err(InputError::WaterExceedsCapacity {
                water_mm: soil_water.get(),
                capacity_mm: soil_water_capacity.get(),
            });
        }
        Ok(Self {
            herbage,
            soil_water,
            soil_water_capacity,
        })
    }
}

/// Daily net forage growth as supplied by a calibrated model or observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DailyForcing {
    pub precipitation: WaterMm,
    pub irrigation: WaterMm,
    pub potential_evapotranspiration: WaterMm,
    pub net_herbage_growth: ForageKgDmPerHa,
    /// Herbage disappearance/removal; not equivalent to animal dry-matter intake.
    pub herbage_removal: ForageKgDmPerHa,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DailyFluxes {
    pub actual_evapotranspiration: WaterMm,
    pub unmet_evapotranspiration: WaterMm,
    pub runoff: WaterMm,
    pub realized_herbage_removal: ForageKgDmPerHa,
    pub unmet_herbage_removal: ForageKgDmPerHa,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResult {
    pub state: PatchState,
    pub fluxes: DailyFluxes,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputError {
    InvalidQuantity { name: &'static str, value: f64 },
    ArithmeticOverflow { name: &'static str },
    WaterExceedsCapacity { water_mm: f64, capacity_mm: f64 },
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuantity { name, value } => {
                write!(f, "{name} must be finite and non-negative, got {value}")
            }
            Self::ArithmeticOverflow { name } => write!(f, "arithmetic overflow in {name}"),
            Self::WaterExceedsCapacity {
                water_mm,
                capacity_mm,
            } => {
                write!(
                    f,
                    "soil water ({water_mm} mm) exceeds capacity ({capacity_mm} mm)"
                )
            }
        }
    }
}

impl std::error::Error for InputError {}

/// Apply one daily water/forage ledger transition.
///
/// Water follows a bucket balance: prior storage + rain + irrigation, limited
/// by capacity; overflow is runoff; actual ET is limited by available water.
/// Forage is prior herbage + supplied growth; removal is limited to available
/// herbage. Deficits are reported explicitly. No biological response is inferred.
pub fn step_patch(state: PatchState, forcing: DailyForcing) -> Result<StepResult, InputError> {
    let water_in = state.soil_water.get() + forcing.precipitation.get() + forcing.irrigation.get();
    if !water_in.is_finite() {
        return Err(InputError::ArithmeticOverflow {
            name: "water input",
        });
    }
    let runoff = (water_in - state.soil_water_capacity.get()).max(0.0);
    let available_water = water_in.min(state.soil_water_capacity.get());
    let actual_et = forcing
        .potential_evapotranspiration
        .get()
        .min(available_water);
    let unmet_et = forcing.potential_evapotranspiration.get() - actual_et;
    let next_water = available_water - actual_et;

    let available_herbage = state.herbage.get() + forcing.net_herbage_growth.get();
    if !available_herbage.is_finite() {
        return Err(InputError::ArithmeticOverflow {
            name: "available herbage",
        });
    }
    let realized_removal = forcing.herbage_removal.get().min(available_herbage);
    let unmet_removal = forcing.herbage_removal.get() - realized_removal;
    let next_herbage = available_herbage - realized_removal;

    Ok(StepResult {
        state: PatchState {
            herbage: ForageKgDmPerHa(next_herbage),
            soil_water: WaterMm(next_water),
            soil_water_capacity: state.soil_water_capacity,
        },
        fluxes: DailyFluxes {
            actual_evapotranspiration: WaterMm(actual_et),
            unmet_evapotranspiration: WaterMm(unmet_et),
            runoff: WaterMm(runoff),
            realized_herbage_removal: ForageKgDmPerHa(realized_removal),
            unmet_herbage_removal: ForageKgDmPerHa(unmet_removal),
        },
    })
}

/// Stable C ABI for scalar daily patch steps. Returns 0 on success and -1 for
/// invalid inputs. Output pointers may be null only when the caller ignores
/// the corresponding output; inputs must point to readable finite values.
/// All numbers use the units named in the parameter suffixes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn campo_step_patch_v1(
    herbage_kg_dm_ha: f64,
    soil_water_mm: f64,
    soil_capacity_mm: f64,
    precipitation_mm: f64,
    irrigation_mm: f64,
    potential_et_mm: f64,
    net_growth_kg_dm_ha: f64,
    requested_removal_kg_dm_ha: f64,
    out_herbage_kg_dm_ha: *mut f64,
    out_soil_water_mm: *mut f64,
    out_runoff_mm: *mut f64,
    out_actual_et_mm: *mut f64,
    out_unmet_removal_kg_dm_ha: *mut f64,
) -> i32 {
    let parsed = (|| {
        let state = PatchState::new(
            ForageKgDmPerHa::new(herbage_kg_dm_ha).ok()?,
            WaterMm::new(soil_water_mm).ok()?,
            WaterMm::new(soil_capacity_mm).ok()?,
        )
        .ok()?;
        let forcing = DailyForcing {
            precipitation: WaterMm::new(precipitation_mm).ok()?,
            irrigation: WaterMm::new(irrigation_mm).ok()?,
            potential_evapotranspiration: WaterMm::new(potential_et_mm).ok()?,
            net_herbage_growth: ForageKgDmPerHa::new(net_growth_kg_dm_ha).ok()?,
            herbage_removal: ForageKgDmPerHa::new(requested_removal_kg_dm_ha).ok()?,
        };
        step_patch(state, forcing).ok()
    })();
    let Some(result) = parsed else { return -1 };

    // SAFETY: each non-null pointer is documented as writable by the C caller.
    unsafe {
        if !out_herbage_kg_dm_ha.is_null() {
            *out_herbage_kg_dm_ha = result.state.herbage.get();
        }
        if !out_soil_water_mm.is_null() {
            *out_soil_water_mm = result.state.soil_water.get();
        }
        if !out_runoff_mm.is_null() {
            *out_runoff_mm = result.fluxes.runoff.get();
        }
        if !out_actual_et_mm.is_null() {
            *out_actual_et_mm = result.fluxes.actual_evapotranspiration.get();
        }
        if !out_unmet_removal_kg_dm_ha.is_null() {
            *out_unmet_removal_kg_dm_ha = result.fluxes.unmet_herbage_removal.get();
        }
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn campo_core_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q_water(v: f64) -> WaterMm {
        WaterMm::new(v).unwrap()
    }
    fn q_forage(v: f64) -> ForageKgDmPerHa {
        ForageKgDmPerHa::new(v).unwrap()
    }

    #[test]
    fn daily_step_is_deterministic_and_closes_ledgers() {
        let state = PatchState::new(q_forage(1200.0), q_water(40.0), q_water(50.0)).unwrap();
        let forcing = DailyForcing {
            precipitation: q_water(20.0),
            irrigation: q_water(2.0),
            potential_evapotranspiration: q_water(6.0),
            net_herbage_growth: q_forage(35.0),
            herbage_removal: q_forage(80.0),
        };
        let a = step_patch(state, forcing).unwrap();
        let b = step_patch(state, forcing).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.state.soil_water.get(), 44.0);
        assert_eq!(a.fluxes.runoff.get(), 12.0);
        assert_eq!(a.state.herbage.get(), 1155.0);
        assert_eq!(
            a.state.soil_water.get()
                + a.fluxes.actual_evapotranspiration.get()
                + a.fluxes.runoff.get(),
            40.0 + 20.0 + 2.0
        );
        assert_eq!(
            a.state.herbage.get() + a.fluxes.realized_herbage_removal.get(),
            1200.0 + 35.0
        );
    }

    #[test]
    fn reports_water_and_herbage_shortfalls() {
        let state = PatchState::new(q_forage(10.0), q_water(1.0), q_water(100.0)).unwrap();
        let forcing = DailyForcing {
            precipitation: q_water(0.0),
            irrigation: q_water(0.0),
            potential_evapotranspiration: q_water(4.0),
            net_herbage_growth: q_forage(0.0),
            herbage_removal: q_forage(14.0),
        };
        let result = step_patch(state, forcing).unwrap();
        assert_eq!(result.state.soil_water.get(), 0.0);
        assert_eq!(result.fluxes.unmet_evapotranspiration.get(), 3.0);
        assert_eq!(result.state.herbage.get(), 0.0);
        assert_eq!(result.fluxes.unmet_herbage_removal.get(), 4.0);
    }

    #[test]
    fn rejects_invalid_initial_states_and_quantities() {
        assert!(WaterMm::new(f64::NAN).is_err());
        assert!(ForageKgDmPerHa::new(-1.0).is_err());
        assert!(PatchState::new(q_forage(1.0), q_water(8.0), q_water(7.0)).is_err());

        let huge = WaterMm::new(f64::MAX).unwrap();
        let state = PatchState::new(q_forage(0.0), huge, huge).unwrap();
        let forcing = DailyForcing {
            precipitation: huge,
            irrigation: q_water(0.0),
            potential_evapotranspiration: q_water(0.0),
            net_herbage_growth: q_forage(0.0),
            herbage_removal: q_forage(0.0),
        };
        assert!(matches!(
            step_patch(state, forcing),
            Err(InputError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn c_abi_returns_state_and_rejects_invalid_input() {
        let (mut herbage, mut soil_water, mut runoff, mut actual_et, mut unmet_removal) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        let status = unsafe {
            campo_step_patch_v1(
                100.0,
                8.0,
                10.0,
                5.0,
                0.0,
                4.0,
                2.0,
                105.0,
                &mut herbage,
                &mut soil_water,
                &mut runoff,
                &mut actual_et,
                &mut unmet_removal,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(herbage, 0.0);
        assert_eq!(soil_water, 6.0);
        assert_eq!(runoff, 3.0);
        assert_eq!(actual_et, 4.0);
        assert_eq!(unmet_removal, 3.0);

        let invalid = unsafe {
            campo_step_patch_v1(
                -1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(invalid, -1);
        assert_eq!(campo_core_abi_version(), 1);
    }

    #[test]
    fn versioned_animal_contract_preserves_bout_memory_and_checks_actions() {
        let observation = ObservationV1 {
            schema_version: OBSERVATION_SCHEMA_V1,
            animal: CowStateV1 {
                animal_id: AnimalId(42),
                body_weight: BodyWeightKg::new(487.5).unwrap(),
            },
            observed_at_minute: 180,
            current_bout: CurrentBoutV1 {
                behavior: BehaviorKindV1::Grazing,
                start_minute: 150,
                elapsed_minutes: 30,
            },
            memory: AgentMemoryV1 {
                recent_bouts: [
                    BoutRecordV1 {
                        behavior: BehaviorKindV1::Resting,
                        start_minute: 90,
                        duration_minutes: 40,
                    },
                    BoutRecordV1 {
                        behavior: BehaviorKindV1::Walking,
                        start_minute: 130,
                        duration_minutes: 20,
                    },
                    BoutRecordV1::default(),
                    BoutRecordV1::default(),
                    BoutRecordV1::default(),
                    BoutRecordV1::default(),
                    BoutRecordV1::default(),
                    BoutRecordV1::default(),
                ],
                len: 2,
            },
        };
        assert!(observation.validate().is_ok());

        let invalid_observation = ObservationV1 {
            observed_at_minute: 181,
            ..observation
        };
        assert_eq!(
            invalid_observation.validate(),
            Err(ContractError::InvalidBoutTime)
        );

        let action = ActionV1 {
            schema_version: ACTION_SCHEMA_V1,
            kind: ActionKindV1::MoveToSite,
            target_site: Some(SiteId(7)),
            requested_duration_minutes: 15,
        };
        assert!(action.validate().is_ok());
        assert_eq!(
            ActionV1 {
                target_site: None,
                ..action
            }
            .validate(),
            Err(ContractError::InvalidActionTarget)
        );
    }
}
