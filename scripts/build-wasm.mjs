import { mkdir, copyFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = resolve(root, 'engine/campo-core/Cargo.toml');
const artifact = resolve(root, 'engine/campo-core/target/wasm32-unknown-unknown/release/campo_core.wasm');
const destination = resolve(root, 'public/engine/campo_core.wasm');

const build = spawnSync('cargo', [
  'build', '--release', '--manifest-path', manifest, '--target', 'wasm32-unknown-unknown',
], { cwd: root, stdio: 'inherit' });

if (build.error) throw build.error;
if (build.status !== 0) process.exit(build.status ?? 1);

await mkdir(dirname(destination), { recursive: true });
await copyFile(artifact, destination);
console.log(`Built Rust WASM core: ${destination}`);
