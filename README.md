# curl_matrix

Rust project that renders curl noise particle animations on RGB LED matrix displays. Targets Raspberry Pi hardware with LED matrix panels (typically 64x64).

Uses a 3D simplex noise implementation with derivative calculation to generate a divergence-free 2D vector field (curl noise) that drives particle motion.

## Workspace Structure

- **`crates/curl-core`** — Core library providing simplex noise, curl noise vector field generation, and generic particle types
- **`crates/curl-matrix`** — Main binary: particle system simulation loop and LED canvas rendering
- **`crates/examples`** — Example binaries (e.g., static image display)

## Build & Run

Cross-compiles to `aarch64-unknown-linux-gnu` (ARM64) for Raspberry Pi.

```bash
cargo build --release
```

**Main binary:**

```bash
cargo run --release --bin curl_matrix -- --particle-count 2000 --field-scale 0.05
```

**Image example:**

```bash
cargo run --release --bin image
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--particle-count` | 2000 | Total number of particles |
| `--field-scale` | 0.05 | Noise coordinate scaling |
| `--particle-speed` | 0.01 | Particle movement scaling |
| `--flow-speed` | 0.09 | Overall flow scaling |
| `--particle-ttl` | 2500 | Max particle lifetime (frames) |
| `--field-update-interval` | 120 | Frames between field recomputations |
| `--time-speed` | 0.01 | Noise field time progression speed |

Standard `rpi-led-matrix` hardware arguments are also supported.

## How It Works

1. `curl-core::generate_vector_field()` pre-computes a vector field over the display dimensions using 3D simplex noise (x, y, time)
2. The curl of the noise gradient produces a divergence-free 2D flow field: `vector = (dy, -dx)`
3. Particles are spawned at random positions with random colors and follow the field
4. Particles wrap around screen edges and regenerate when their TTL expires
5. The vector field is recomputed at a configurable interval to animate the flow

## Cross-Compilation

Configured in `.cargo/config.toml`:
- Target: `aarch64-unknown-linux-gnu`
- Linker: `/usr/bin/aarch64-linux-gnu-gcc`

The `rpi-led-matrix` dependency is a local path reference to `rust-rpi-rgb-led-matrix/rpi-led-matrix`.

## Dependencies

- `rpi-led-matrix` — Rust bindings for the rpi-rgb-led-matrix C++ library
- `nalgebra` — Linear algebra (`Point2`, `Point3`, `Vector2`, `Vector3`)
- `clap` — CLI argument parsing
- `rand` — Random number generation
- `embedded-graphics` — Used by the image example

## Roadmap

### Tier 1 — High Impact, Straightforward

- **Fade / Trail Effect** — Persist the frame buffer between frames and dim it each tick instead of clearing to black. Produces luminous particle trails. CLI arg: `--fade` (0.0–1.0). Requires a software frame buffer (`Vec<LedColor>`) that persists across frames.

- **Color Palettes** — Replace random RGB with curated palettes (fire, ocean, aurora, monochrome). New `palette` module in `curl-core` with anchor-color interpolation. CLI arg: `--palette <name>`.

- **Speed-Based Brightness** — Modulate particle brightness by velocity magnitude. Fast particles are bright, slow ones are dim.

- **Particle Size Variation** — Particles can be 1x1, 2x2, or 3x3 pixels (weighted random at spawn) for visual depth and density variation.

### Tier 2 — High Impact, Moderate Effort

- **Multi-Layer Noise (Octave Curl)** — Stack multiple noise layers at different frequencies/amplitudes for organic, detailed flow patterns. CLI args: `--octaves` (1–4), `--persistence` (0.0–1.0). Wraps existing simplex calls — no changes to `simplex.rs`.

- **Velocity-Responsive Color Mapping** — Color particles by velocity direction (angle → hue) and magnitude (speed → brightness). Creates visible flow lines. Pre-compute a color field alongside the vector field.

- **Runtime Parameter Control** — Change parameters at runtime via stdin commands (e.g., `palette ocean`, `fade 0.85`) or Unix signals, without restarting.

- **Configuration File Support** — Load/save presets from TOML files. CLI args override file values. Adds `serde` + `toml` dependencies.

### Tier 3 — Nice-to-Have Polish

- **FPS Counter** — Log frame rate and field-update time to stderr for tuning on the Pi.
- **Time-Varying Flow Direction** — Flow vector rotates via `cos`/`sin` over time for more variety.
- **Particle Spawn Patterns** — Edge-only, center-burst, or ring spawn modes instead of uniform random.
- **Gamma / Brightness Control** — LUT-based gamma correction to compensate for LED non-linear brightness response.
- **Symmetry Modes** — 4/6/8-fold kaleidoscopic mirroring of the noise field for mandala-like patterns.

## License

MIT
