"""Replay osmnx.graph_from_polygon offline on the cached Overpass response for a candidate polygon
(valid as long as the candidate lies within the polygon the cache was downloaded for, plus 500 m)."""
import os, sys, re, glob, json, time
import geopandas as gpd, networkx as nx, osmnx as ox
from osmnx import truncate, simplification, projection
from osmnx.graph import _create_graph
poly = gpd.read_file(sys.argv[1]).geometry.union_all()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'extract_data.py')).read()
exit_nodes = [int(x) for x in re.findall(r'\b\d{4,}\b', re.search(r'exit_nodes = \[(.*?)\n\]', src, re.S).group(1))]
t = time.time()
resp = json.load(open(glob.glob('osmnx_cache/*.json')[0]))
poly_proj, crs = projection.project_geometry(poly)
poly_buff, _ = projection.project_geometry(poly_proj.buffer(500), crs=crs, to_latlong=True)
G = _create_graph([resp], False)
G = truncate.truncate_graph_polygon(G, poly_buff, truncate_by_edge=False)
G = truncate.largest_component(G, strongly=False)
Gb = G
G = simplification.simplify_graph(Gb)
G = truncate.truncate_graph_polygon(G, poly, truncate_by_edge=False)
G = truncate.largest_component(G, strongly=False)
print(f"[{time.time()-t:.0f}s] nodes: {G.number_of_nodes():,} (thesis 46,290)  edges: {G.number_of_edges():,} (thesis 104,026)")
missing = [n for n in exit_nodes if n not in G]
print(f"exit nodes present: {len(exit_nodes)-len(missing)}/{len(exit_nodes)}  missing: {missing}")
for n in missing:
    if n not in Gb: print(f"  {n}: not in buffered graph at all"); continue
    nb = set(Gb.predecessors(n)) | set(Gb.successors(n))
    print(f"  {n}: in buffered graph, in={Gb.in_degree(n)} out={Gb.out_degree(n)} neighbours={len(nb)} -> simplified away (pass-through node)")
if len(sys.argv) > 2: ox.save_graphml(G, sys.argv[2])
