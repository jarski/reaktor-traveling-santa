#!/usr/bin/env python3
"""Independent validator + scorer for a Traveling Santa solution CSV.

Checks: every child served exactly once, no route over the 10 t capacity.
Reports total great-circle distance in metres (sphere, R = 6378 km).
Accepts ';' or ',' separators, so it reads every solver's output format.

    python3 tools/score.py <solution> [nicelist.txt]
"""
import math
import os
import sys

R_KM = 6378.0
CAP = 10_000_000
DEPOT = (68.073611, 29.315278)


def unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def gc(a, b):
    d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 2.0 * R_KM * math.asin(min(1.0, d / 2.0))


DEFAULT_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nicelist.txt")


def main(path, listfile=DEFAULT_LIST):
    pts, dem = {}, {}
    for line in open(listfile):
        line = line.strip()
        if not line:
            continue
        i, lat, lon, w = line.split(";")
        pts[int(i)] = unit(float(lat), float(lon))
        dem[int(i)] = int(w)
    depot = unit(*DEPOT)

    seen, total, nroutes, maxload, over = set(), 0.0, 0, 0, 0
    dup = []
    for line in open(path):
        ids = [int(t) for t in line.replace(",", ";").split(";") if t.strip()]
        if not ids:
            continue
        nroutes += 1
        load = sum(dem[i] for i in ids)
        maxload = max(maxload, load)
        if load > CAP:
            over += 1
        for i in ids:
            if i in seen:
                dup.append(i)
            seen.add(i)
        prev = depot
        for i in ids:
            total += gc(prev, pts[i])
            prev = pts[i]
        total += gc(prev, depot)

    missing = set(pts) - seen
    ok = not missing and not dup and not over
    print(f"routes        : {nroutes}")
    print(f"served        : {len(seen)} / {len(pts)}")
    print(f"duplicates    : {len(dup)}")
    print(f"missing       : {len(missing)}")
    print(f"capacity viol : {over}   (max load {maxload} g)")
    print(f"fill rate     : {sum(dem.values()) / (nroutes * CAP) * 100:.2f}%")
    print(f"TOTAL         : {total * 1000:,.0f} m   ({total:,.1f} km)")
    print(f"VALID         : {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
