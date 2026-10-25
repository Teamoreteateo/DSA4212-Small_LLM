"""Synthetic story corpus for the smoke test (never used in the report)."""
import os, random, sys
out = sys.argv[1] if len(sys.argv) > 1 else "data_smoke/corpus.txt"
rng = random.Random(0)
names = ["Lily", "Tom", "Mia", "Ben", "Sam", "Anna", "Max", "Zoe"]
things = ["ball", "kite", "cat", "dog", "box", "cake", "tree", "boat"]
feels = ["happy", "sad", "scared", "excited", "tired", "proud"]
places = ["park", "garden", "beach", "forest", "house", "school"]
stories = []
for _ in range(6000):
    n1, n2 = rng.sample(names, 2); t, f, p = rng.choice(things), rng.choice(feels), rng.choice(places)
    stories.append(f"One day, {n1} went to the {p}. {n1} saw a big {t}. {n1} felt {f}. "
                   f"{n2} came and said, \"Can I play with the {t}?\" {n1} said yes, and they played all day. The end.")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w").write("\n\n".join(stories))
print(f"wrote {len(stories)} stories to {out}")