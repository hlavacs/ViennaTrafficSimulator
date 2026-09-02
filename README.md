# Vienna Traffic Simulator

GPU-accelerated evacuation simulation of Vienna's road network. Roughly 680,000 vehicles are
spawned from OpenStreetMap building footprints and parking areas and routed to 125 perimeter exits
with car-following physics (Intelligent Driver Model) running in Vulkan compute shaders written in
Slang. A multi-threaded CPU engine implements the same model for comparison. The simulator was
developed for the bachelor thesis *Vienna Traffic Simulator: A GPU-Accelerated Urban Traffic
Model* by Denys-Lev Gusti (University of Vienna, 2026, supervised by Helmut Hlavacs).

## Repository contents

| Path | Purpose |
|---|---|
| `main.cpp`, `common.hpp`, `cpu_engine.hpp` | Simulator: Vulkan setup, GPU and CPU engines, Dear ImGui GUI, headless mode |
| `shaders/*.slang` | Compute shaders (edge reset, spatial grid, physics) and rendering shaders |
| `vulkan_nodes.bin`, `vulkan_edges.bin`, `vulkan_cars.bin` | Prebuilt road network and vehicle fleet, loaded directly into GPU buffers |
| `config.json` | Runtime configuration (see below) |
| `extract_data.py` | Regenerates the three `.bin` files from OpenStreetMap |
| `vienna_combined.geojson` | Study-area polygon used by `extract_data.py` |
| `tools/polygon_reconstruction/` | Scripts that rebuilt `vienna_combined.geojson` and check that every exit node survives graph simplification |

The committed `.bin` files are the dataset the simulator expects, so building and running does not
require Python or network access. Regenerate them only if you change the polygon, the exit nodes, or
want fresher OSM data (results will differ slightly because OSM changes continuously).

## Prerequisites

- CMake 3.28 or newer and a C++23 compiler (GCC 14+ or Clang 18+; `std::print` is used)
- Vulkan headers and loader (`libvulkan-dev` on Debian/Ubuntu, or the LunarG SDK)
- `slangc`, the [Slang](https://github.com/shader-slang/slang) shader compiler, on the `PATH`
  (ships with the LunarG Vulkan SDK, or download a release tarball and add its `bin/` directory)
- Internet access on first configure: GLFW 3.4, Dear ImGui (docking branch) and nlohmann/json are
  fetched with CMake's FetchContent
- A Vulkan 1.3 capable GPU with timestamp query support for GPU mode; the CPU engine needs no GPU

## Build

```bash
export PATH=/path/to/slang/bin:$PATH      # if slangc is not already on the PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The build compiles the shaders to SPIR-V and copies `config.json` and the `.bin` files next to the
executable in `build/`.

## Run

The executable resolves the data paths in `config.json` relative to the working directory, so run
it from `build/`:

```bash
cd build
./BA                      # uses ./config.json
./BA path/to/other.json   # explicit configuration
```

`config.json` keys:

| Key | Meaning |
|---|---|
| `vulkan_nodes_path`, `vulkan_edges_path`, `vulkan_cars_path` | Input files |
| `headless` | `true` runs without a window and exits when the evacuation is complete |
| `use_gpu` | `true` for the Vulkan compute engine, `false` for the CPU engine |
| `cpu_threads` | Worker threads for the CPU engine, `0` = all hardware threads |
| `output_file` | JSON results written on exit |
| `participation` | Fraction of the fleet that takes part (`1.0` = every car) |
| `closed_exits` | Exit indices that are closed from the start |

In the GUI, space pauses and the arrow keys change the simulation speed; exits can be opened and
closed at runtime, which triggers a Dijkstra recalculation of the flow field. Console output is
block-buffered when redirected, so use `stdbuf -oL ./BA ...` to watch headless progress in a log.

### Selecting the GPU

The simulator uses the first Vulkan device the loader reports. On a laptop with an integrated GPU
that is usually the wrong one; restrict the loader to a single driver instead:

```bash
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./BA
```

## Regenerating the data

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python extract_data.py
```

The script downloads the road network and building/parking features inside
`vienna_combined.geojson` from the official Overpass API, computes flow-field routes to the exit
nodes, estimates parking capacity per edge, and writes the three `.bin` files plus GeoJSON
intermediates for inspection in QGIS. It takes about ten minutes and needs a few hundred megabytes
of RAM. Every ID in the `exit_nodes` list must survive OSMnx's graph simplification, otherwise the
Dijkstra step aborts; `tools/polygon_reconstruction/README.md` explains how to check and fix that
without re-downloading.

## Binary layout

All three files are flat arrays of little-endian structs matching `common.hpp`:

- node, 16 bytes: `float x, y; int32 lock; int32 type` (0 normal, 1 open exit, 2 closed exit, 3 blind)
- edge, 32 bytes: `int32 start, end, next_edge; float length, max_speed; int32 head_car, garage_lock, spawn_capacity`
- car, 32 bytes: `int32 edge; float position, speed; int32 state, next_car; int32 pad[3]`
