import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const wasmPath = new URL('../target/wasm32-unknown-unknown/release/campo_core.wasm', import.meta.url);
const bytes = await readFile(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes);
const { campo_core_abi_version, campo_step_patch_v1 } = instance.exports;

assert.equal(campo_core_abi_version(), 1);
assert.equal(campo_step_patch_v1(100, 8, 10, 5, 0, 4, 2, 105, 0, 0, 0, 0, 0), 0);
assert.equal(campo_step_patch_v1(-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), -1);
console.log('WASM ABI v1 instantiated; valid and invalid calls returned expected status.');
