"""Split a session profile's time into physics, planning, rendering, recording, skills and waits."""
import pstats
import sys

stats = pstats.Stats(sys.argv[1])
total = max(v[3] for v in stats.stats.values())  # the outermost cumulative time
print(f"total profiled {total:.0f} s\n")
print("top 30 by cumulative time:")
rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][3])
for (file, line, name), (cc, nc, tt, ct, callers) in rows[:30]:
    print(f"{ct:8.1f} cum {tt:8.1f} self {nc:9d} calls  {file.split('site-packages/')[-1].split('src/')[-1]}:{line} {name}")
print("\ntop 25 by own time:")
for (file, line, name), (cc, nc, tt, ct, callers) in sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:25]:
    print(f"{tt:8.1f} self {nc:9d} calls  {file.split('site-packages/')[-1].split('src/')[-1]}:{line} {name}")
