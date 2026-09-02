# Reconstructing `vienna_combined.geojson`

The original study-area polygon was never committed. The thesis (section 4.1) describes it as
Vienna's official border "expanded slightly past Vienna's official borders to keep major perimeter
highways intact". These scripts rebuild an equivalent polygon from the 125 exit node IDs hard-coded
in `extract_data.py`, which must all survive `ox.graph_from_polygon` (osmnx simplifies the graph on
the polygon padded by 500 m, then clips, so a node survives only as a dead end or intersection there).

Inputs (already included):
- `vienna_nominatim.json`: OSM relation 109166 (Wien) boundary from Nominatim (`lookup?osm_ids=R109166&polygon_geojson=1`).
- `exit_nodes.json`: coordinates of the exit nodes from Overpass (`node(id:...);out;`).
- an osmnx cache directory `osmnx_cache/` holding one network download (same highway filter as
  `extract_data.py`) for a generous polygon (Vienna + 300 m); create it by running
  `ox.graph_from_polygon` once with `ox.settings.use_cache = True`, `ox.settings.cache_folder = "osmnx_cache"`.

Steps (run from this directory with the project venv):

    python build_polygon.py 15 20 40 vienna_combined.geojson   # base buffer, corridor width, inward margin (m)
    python audit_exits.py vienna_combined.geojson              # fast topological check of each exit
    python simulate.py vienna_combined.geojson                 # replays osmnx's pipeline offline on the cache
    python trace_simplified.py vienna_combined.geojson <node>  # diagnose one exit

Result on 2026-09-02 (osmnx 2.1.1): 125/125 exits present, 46,125 nodes and 103,586 edges
versus 46,290 / 104,026 in the thesis.
