"""For a candidate polygon P, check each exit node's raw topology in the cached network:
osmnx keeps a node through simplification only if it is a dead end or has != 2 neighbours
(or degree not in {2,4}) within P+500m; and it must keep an inside neighbour after clipping to P."""
import json, math, sys, glob
from collections import defaultdict
from shapely.geometry import shape, Point
from shapely.ops import transform
from shapely.prepared import prep
KX=111320.0*math.cos(math.radians(48.2)); KY=110540.0
fwd=lambda x,y,z=None:(x*KX,y*KY)
P=transform(fwd, shape(json.load(open(sys.argv[1]))['features'][0]['geometry'])); P500=P.buffer(500)
Pp, P500p = prep(P), prep(P500)
vienna=transform(fwd, shape(json.load(open('vienna_nominatim.json'))[0]['geojson']))
exits=[e['id'] for e in json.load(open('exit_nodes.json'))['elements']]
els=json.load(open(glob.glob('osmnx_cache/*.json')[0]))['elements']
nodes={e['id']:fwd(e['lon'],e['lat']) for e in els if e['type']=='node'}
succ=defaultdict(set); pred=defaultdict(set); ways_of=defaultdict(list)
for w in els:
    if w['type']!='way': continue
    ow=w.get('tags',{}).get('oneway','no'); ids=w['nodes']
    for a,b in zip(ids, ids[1:]):
        if a not in nodes or b not in nodes: continue
        ways_of[a].append(w['id']); ways_of[b].append(w['id'])
        if ow in ('yes','true','1'): succ[a].add(b); pred[b].add(a)
        elif ow=='-1': succ[b].add(a); pred[a].add(b)
        else: succ[a].add(b); pred[b].add(a); succ[b].add(a); pred[a].add(b)
def endpoint(n, keep):
    s={v for v in succ[n] if v in keep}; p={v for v in pred[n] if v in keep}
    nb=s|p; deg=len(s)+len(p)
    if n in nb: return True, "self-loop"
    if not s or not p: return True, "dead end"
    if not (len(nb)==2 and deg in (2,4)): return True, f"intersection n={len(nb)} d={deg}"
    return False, f"pass-through n={len(nb)} d={deg}"
only = set(int(x) for x in sys.argv[2:]) if len(sys.argv)>2 else None
bad=0
for e in exits:
    if only and e not in only: continue
    pt=Point(nodes[e]); nb_all=succ[e]|pred[e]
    keep500={v for v in nb_all if P500p.contains(Point(nodes[v]))}|{e}
    ok, why = endpoint(e, keep500)
    inside_nb=[v for v in nb_all if Pp.contains(Point(nodes[v]))]
    verdict = "OK" if ok and inside_nb else "FAIL"
    if verdict=="FAIL" or only:
        bad += verdict=="FAIL"
        print(f"{verdict} {e}: {pt.distance(vienna.exterior):.0f} m {'in' if vienna.contains(pt) else 'OUT of'} Vienna; margin in P {pt.distance(P.exterior)*(1 if Pp.contains(pt) else -1):.0f} m; {why}; raw neighbours {len(nb_all)}, inside P: {len(inside_nb)}")
        for v in sorted(nb_all, key=lambda v: math.dist(nodes[e],nodes[v])):
            q=Point(nodes[v]); print(f"      nb {v}: {math.dist(nodes[e],nodes[v]):5.0f} m away, {'in P' if Pp.contains(q) else f'{q.distance(P.exterior):.0f} m outside P'}, {'in' if P500p.contains(q) else 'OUT of'} P+500; ways {ways_of[v][:3]}")
print(f"\n{bad} exits predicted to fail with {sys.argv[1]}")
