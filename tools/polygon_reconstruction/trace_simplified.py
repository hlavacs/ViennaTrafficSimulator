"""Walk from an exit along each road through pass-through nodes (osmnx endpoint rule within P+500)
until the next endpoint, reporting whether that endpoint is inside P (i.e. survives final clipping)."""
import json, math, sys, glob
from collections import defaultdict
from shapely.geometry import shape, Point
from shapely.ops import transform
from shapely.prepared import prep
KX=111320.0*math.cos(math.radians(48.2)); KY=110540.0
fwd=lambda x,y,z=None:(x*KX,y*KY)
P=transform(fwd, shape(json.load(open(sys.argv[1]))['features'][0]['geometry'])); P500=P.buffer(500)
Pp, P500p = prep(P), prep(P500)
els=json.load(open(glob.glob('osmnx_cache/*.json')[0]))['elements']
nodes={e['id']:fwd(e['lon'],e['lat']) for e in els if e['type']=='node' and P500p.contains(Point(fwd(e['lon'],e['lat'])))}
succ=defaultdict(set); pred=defaultdict(set); tag={}
for w in els:
    if w['type']!='way': continue
    ow=w.get('tags',{}).get('oneway','no'); ids=w['nodes']
    for a,b in zip(ids, ids[1:]):
        if a not in nodes or b not in nodes: continue
        tag[(a,b)]=tag[(b,a)]=f"{w.get('tags',{}).get('highway')}/{w.get('tags',{}).get('name')}"
        if ow in ('yes','true','1'): succ[a].add(b); pred[b].add(a)
        elif ow=='-1': succ[b].add(a); pred[a].add(b)
        else: succ[a].add(b); pred[b].add(a); succ[b].add(a); pred[a].add(b)
def is_endpoint(n):
    nb=succ[n]|pred[n]; deg=len(succ[n])+len(pred[n])
    return n in nb or not succ[n] or not pred[n] or not (len(nb)==2 and deg in (2,4))
for e in [int(x) for x in sys.argv[2:]]:
    print(f"exit {e}: endpoint={is_endpoint(e)}, in P={Pp.contains(Point(nodes[e]))}")
    for first in succ[e]|pred[e]:
        path=[e, first]; cur=first
        while not is_endpoint(cur):
            nxt=[v for v in succ[cur]|pred[cur] if v!=path[-2]]
            if len(nxt)!=1: break
            cur=nxt[0]; path.append(cur)
        q=Point(nodes[cur]); inside=[Pp.contains(Point(nodes[n])) for n in path]
        print(f"   via {first} ({tag.get((e,first))}): simplified neighbour {cur} after {len(path)-1} hops, "
              f"{'INSIDE P' if Pp.contains(q) else f'OUTSIDE P by {q.distance(P.exterior):.0f} m'}; path nodes inside P: {sum(inside)}/{len(inside)}")
