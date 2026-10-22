import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const wasmPath = new URL('../target/wasm32-unknown-unknown/release/campo_core.wasm', import.meta.url);
const bytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes);
const {
  campo_core_abi_version,
  campo_step_patch_v1,
  campo_estimate_water_intake_brcorte_2023_v1,
  memory,
  __heap_base,
} = instance.exports;

assert.equal(campo_core_abi_version(), 1);
const scratch = Math.ceil(Number(__heap_base.value) / 8) * 8;
const status = campo_step_patch_v1(100, 8, 10, 5, 0, 4, 2, 105,
  scratch, scratch + 8, scratch + 16, scratch + 24, scratch + 32);
assert.equal(status, 0);
const values = new DataView(memory.buffer);
assert.equal(values.getFloat64(scratch, true), 0);
assert.equal(values.getFloat64(scratch + 8, true), 6);
assert.equal(values.getFloat64(scratch + 16, true), 3);
assert.equal(values.getFloat64(scratch + 24, true), 4);
assert.equal(values.getFloat64(scratch + 32, true), 3);
assert.equal(campo_step_patch_v1(-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), -1);
const waterOutput = scratch + 40;
const domainOutput = scratch + 48;
assert.equal(campo_estimate_water_intake_brcorte_2023_v1(350, 7, 30, 60, waterOutput, domainOutput), 0);
assert.ok(Math.abs(values.getFloat64(waterOutput, true) - 20.8366286267) < 1e-8);
assert.equal(values.getUint32(domainOutput, true), 1);
assert.equal(campo_estimate_water_intake_brcorte_2023_v1(350, 7, 38, 60, waterOutput, domainOutput), 0);
assert.equal(values.getUint32(domainOutput, true), 0);
assert.equal(campo_estimate_water_intake_brcorte_2023_v1(350, 7, 30, 101, waterOutput, domainOutput), -1);
console.log('WASM ABI v1 patch ledger and BR-CORTE water equation verified.');
