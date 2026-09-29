# Protein Folding

Four interactive 3D protein-folding searches compete in fixed 30-second rounds. Each starts with the same 28-residue hydrophobic/polar chain and uses an independent random sequence. The lowest best energy wins; the best conformations are held for four seconds before the next round. Scores persist across rounds. Pausing or hiding the page pauses the round clock.

WebGPU runs four workgroups in parallel. Each workgroup evolves one chain and distributes its 351 nonbonded pair-energy evaluations across 64 lanes, reducing the result before accepting or rejecting each Metropolis pivot. Current and best conformations remain on the GPU. Embedded Swift, compiled to WebAssembly, initializes the chains and independently scores the saved best conformations before choosing the winner. GPU kernels use WGSL; JavaScript manages browser APIs and the rotatable view. WebGPU is required for the four-way race. The original Swift CPU solver remains available for numerical tests.

## Run

Serve this directory over localhost or HTTPS and open index.html. The compiled core.wasm is included. Click Run simulation, drag to rotate, scroll to zoom, or double-click to reset the camera. Execution pauses when the page is hidden.

## Build and test

Use the official Swift 6.4 toolchain and matching WebAssembly SDK.

```sh
export SWIFT_BIN=/path/to/swift-toolchain/usr/bin
export WASI_SYSROOT=/path/to/swift-wasm-sdk/wasm32-unknown-wasip1/WASI.sdk
node build.mjs
node test.mjs
node test-rounds.mjs
```

Tests verify energy reduction, bond-length conservation, round cutoff, waiting for pending GPU work, lowest-energy winner selection, pause semantics, and round restart.

Bundled dependency licenses are in THIRD_PARTY_NOTICES.md.
