export interface PatchLedgerInput {
  herbageKgDmPerHa: number;
  soilWaterMm: number;
  soilCapacityMm: number;
  precipitationMm: number;
  irrigationMm: number;
  potentialEtMm: number;
  netHerbageGrowthKgDmPerHa: number;
  requestedHerbageRemovalKgDmPerHa: number;
}

export interface PatchLedgerResult {
  herbageKgDmPerHa: number;
  soilWaterMm: number;
  runoffMm: number;
  actualEtMm: number;
  unmetHerbageRemovalKgDmPerHa: number;
}

interface CampoCoreExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  campo_core_abi_version: () => number;
  campo_step_patch_v1: (
    herbage: number,
    soilWater: number,
    soilCapacity: number,
    precipitation: number,
    irrigation: number,
    potentialEt: number,
    netGrowth: number,
    requestedRemoval: number,
    outHerbage: number,
    outSoilWater: number,
    outRunoff: number,
    outActualEt: number,
    outUnmetRemoval: number,
  ) => number;
  __heap_base: WebAssembly.Global;
}

export interface ScienceCore {
  readonly abiVersion: number;
  stepPatch(input: PatchLedgerInput): PatchLedgerResult;
}

export async function loadScienceCore(baseUrl = import.meta.env.BASE_URL): Promise<ScienceCore> {
  const response = await fetch(`${baseUrl}engine/campo_core.wasm`);
  if (!response.ok) throw new Error(`Could not load Campo Vivo WASM core (${response.status})`);
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes);
  const exports = instance.exports as CampoCoreExports;
  const abiVersion = exports.campo_core_abi_version();
  if (abiVersion !== 1) throw new Error(`Unsupported Campo Vivo core ABI: ${abiVersion}`);

  // This first core has no heap allocator or dynamic allocations. Reserve a small
  // aligned result block immediately above __heap_base for synchronous FFI calls.
  const scratch = Math.ceil(Number(exports.__heap_base.value) / 8) * 8;
  const resultBytes = Float64Array.BYTES_PER_ELEMENT * 5;
  if (scratch + resultBytes > exports.memory.buffer.byteLength) {
    const missingPages = Math.ceil((scratch + resultBytes - exports.memory.buffer.byteLength) / 65_536);
    exports.memory.grow(missingPages);
  }

  const view = new DataView(exports.memory.buffer);
  return {
    abiVersion,
    stepPatch(input) {
      const args = [
        input.herbageKgDmPerHa,
        input.soilWaterMm,
        input.soilCapacityMm,
        input.precipitationMm,
        input.irrigationMm,
        input.potentialEtMm,
        input.netHerbageGrowthKgDmPerHa,
        input.requestedHerbageRemovalKgDmPerHa,
      ] as const;
      if (args.some((value) => !Number.isFinite(value) || value < 0)) {
        throw new RangeError('Patch ledger inputs must be finite, non-negative physical quantities');
      }

      const status = exports.campo_step_patch_v1(
        ...args,
        scratch,
        scratch + 8,
        scratch + 16,
        scratch + 24,
        scratch + 32,
      );
      if (status !== 0) throw new RangeError('Rust patch ledger rejected an invalid state or forcing');

      return {
        herbageKgDmPerHa: view.getFloat64(scratch, true),
        soilWaterMm: view.getFloat64(scratch + 8, true),
        runoffMm: view.getFloat64(scratch + 16, true),
        actualEtMm: view.getFloat64(scratch + 24, true),
        unmetHerbageRemovalKgDmPerHa: view.getFloat64(scratch + 32, true),
      };
    },
  };
}
