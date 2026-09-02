"""vienna_combined.geojson v3 = Vienna boundary + BASE buffer
  + for every exit node: a corridor (width CORR) along the real road path until a node >= MARGIN_IN
    inside the base polygon (keeps outlying exits connected and border-hugging exits from being islanded)
  + for exits whose intersection status depends on a raw neighbour further than 500 m from the
    polygon: a short extension toward that neighbour so it lands within polygon+500 m
Road geometry comes from the cached Overpass response of the earlier download."""
import json, math, sys, glob, heapq
from collections import defaultdict
from shapely.geometry import shape, Point, LineString, Polygon, mapping
from shapely.ops import transform, unary_union
from shapely.prepared import prep
BASE, CORR, MARGIN_IN, OUT = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
KX = 111320.0*math.cos(math.radians(48.2)); KY = 110540.0
fwd = lambda x, y, z=None: (x*KX, y*KY); inv = lambda x, y, z=None: (x/KX, y/KY)
vienna = transform(fwd, shape(json.load(open('vienna_nominatim.json'))[0]['geojson']))
exits = [e['id'] for e in json.load(open('exit_nodes.json'))['elements']]
els = json.load(open(glob.glob('osmnx_cache/*.json')[0]))['elements']
nodes = {e['id']: fwd(e['lon'], e['lat']) for e in els if e['type']=='node'}
adj = defaultdict(list)
for w in els:
    if w['type']!='way': continue
    for a, b in zip(w['nodes'], w['nodes'][1:]):
        if a in nodes and b in nodes:
            d = math.dist(nodes[a], nodes[b]); adj[a].append((b, d)); adj[b].append((a, d))
B = vienna.buffer(BASE); Bp = prep(B); Bin = B.buffer(-MARGIN_IN); Binp = prep(Bin)
parts = [B]; stats = defaultdict(int)
for e in exits:
    p = Point(nodes[e]); parts.append(p.buffer(CORR))
    # (1) corridor along road to a node comfortably inside the base polygon
    dist = {e: 0.0}; prev = {}; heap = [(0.0, e)]; found = None
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        if Binp.contains(Point(nodes[u])): found = u; break
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e18): dist[v] = d + w; prev[v] = u; heapq.heappush(heap, (d + w, v))
    if found is None: print(f"  !! no road path from exit {e} into the city"); continue
    path = [found]
    while path[-1] != e: path.append(prev[path[-1]])
    if len(path) > 1:
        parts.append(LineString([nodes[n] for n in path]).buffer(CORR))
        kind = "outside" if not Bp.contains(p) else "border"
        stats[kind] += 1
        if dist[found] > 150: print(f"  corridor {e}: {p.distance(vienna.exterior):4.0f} m {'outside' if not vienna.contains(p) else 'inside'} Vienna, {dist[found]:4.0f} m of road to {MARGIN_IN:.0f} m inside")
    # (2) intersection status: need >= 3 raw neighbours within polygon+500 m
    nb = {v for v, _ in adj[e]}
    if len(nb) >= 3:
        near = {v for v in nb if Point(nodes[v]).distance(B) < 480}
        for v in sorted(nb - near, key=lambda v: math.dist(nodes[e], nodes[v])):
            if len(near) >= 3: break
            L = math.dist(nodes[e], nodes[v]); cut = max(0.0, L - 430.0) / L
            seg = LineString([nodes[e], (nodes[e][0] + (nodes[v][0]-nodes[e][0])*cut, nodes[e][1] + (nodes[v][1]-nodes[e][1])*cut)])
            parts.append(seg.buffer(CORR)); near.add(v); stats["reach"] += 1
            print(f"  reach {e} -> neighbour {v}: {L:.0f} m away, extension {L*cut:.0f} m")
P = unary_union(parts)
if P.geom_type == 'MultiPolygon': P = max(P.geoms, key=lambda g: g.area)
P = Polygon(P.exterior).simplify(1.0)
margins = [Point(nodes[e]).distance(P.exterior) * (1 if P.contains(Point(nodes[e])) else -1) for e in exits]
print(f"base {BASE:.0f} m, corridor {CORR:.0f} m, margin-in {MARGIN_IN:.0f} m; corridors: {dict(stats)}; exits inside {sum(m>0 for m in margins)}/{len(exits)}, min margin {min(margins):.1f} m")
print(f"area {P.area/1e6:.1f} km2 (Vienna {vienna.area/1e6:.1f}), vertices {len(P.exterior.coords)}")
fc = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"Vienna study area (reconstructed v3)","base_buffer_m":BASE,"corridor_width_m":CORR,"margin_in_m":MARGIN_IN},"geometry":mapping(transform(inv, P))}]}
json.dump(fc, open(OUT, 'w')); print("wrote", OUT)
