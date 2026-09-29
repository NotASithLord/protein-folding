# Protein Folding

An interactive 3D protein-folding example, built in Embedded Swift and compiled to WebAssembly. A 28-residue hydrophobic/polar chain uses Metropolis pivot moves to conserve bond lengths, penalize collisions, and favor hydrophobic contacts. The view shows the actual solver coordinates.

## Run

Serve this directory over localhost or HTTPS and open index.html. The compiled core.wasm is included. Click Run simulation, drag to rotate, scroll to zoom, or double-click to reset the camera. Execution pauses when the page is hidden.

## Build and test

Use the official Swift 6.4 toolchain and matching WebAssembly SDK.

```sh
export SWIFT_BIN=/path/to/swift-toolchain/usr/bin
export WASI_SYSROOT=/path/to/swift-wasm-sdk/wasm32-unknown-wasip1/WASI.sdk
node build.mjs
node test.mjs
```

Tests verify energy reduction and bond-length conservation.

Bundled dependency licenses are in THIRD_PARTY_NOTICES.md.
