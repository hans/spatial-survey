"""
Parse the Visual Genome objects JSON file into a simple usable format for our
purposes.
"""


from argparse import ArgumentParser
from collections import Counter
import itertools
import json
import re
import sys

import numpy as np


def iter_scenes(corpus_path):
    with open(corpus_path, "r") as corpus_f:
        vg = json.load(corpus_f)
        for scene in vg:
            objects = []
            for s_object in scene["objects"]:
                synsets = s_object["synsets"]
                if len(synsets) > 0:
                    objects.append(synsets[0])

            yield set(objects)


def iter_relations(relations_path, reln_matcher):
    with open(relations_path, "r") as relations_f:
        relations_d = json.load(relations_f)
        for scene in relations_d:
            if not any(reln_matcher.match(reln["predicate"])
                       for reln in scene["relationships"]):
                continue

            # For now, just extract the relevant relations.
            filtered = [reln for reln in scene["relationships"]
                        if reln_matcher.match(reln["predicate"])]

            for reln in filtered:
                yield featurize_relation(reln, scene)


def featurize_relation(reln, scene):
    return {
        "predicate": reln["predicate"],
        "dist": np.sqrt((reln["subject"]["x"] - reln["object"]["x"]) ** 2 +
                        (reln["subject"]["y"] - reln["object"]["y"]) ** 2)
    }


def main(args):
    matcher = re.compile("next to|near")
    relns = iter_relations(args.relations_path, matcher)

    dists = {"near": [], "next to": []}
    for reln in relns:
        key = matcher.search(reln["predicate"]).group(0)
        dists[key].append(reln["dist"])

    print("near", np.mean(dists["near"]), np.var(dists["near"]))
    print("next to", np.mean(dists["next to"]), np.var(dists["next to"]))

    # scenes = list(iter_scenes(args.corpus_path))

    # if args.freq_threshold > 0:
    #     c = Counter()
    #     for scene in scenes:
    #         c.update(scene)

    #     filtered_objs = set(k for k in c if c[k] >= args.freq_threshold)
    #     sys.stderr.write("Retaining %i of %i objects.\n"
    #                      % (len(filtered_objs), len(c)))
    #     for obj, count in c.most_common(25):
    #         sys.stderr.write("\t%s\t%i\n" % (obj, count))

    #     # Filter out low-freq items.
    #     filtered_scenes = []
    #     for scene in scenes:
    #         filtered_scene = scene & filtered_objs
    #         if len(filtered_scene) > 1:
    #             filtered_scenes.append(filtered_scene)
    #     scenes = filtered_scenes

    # scenes = list(map(list, scenes))
    # json.dump(scenes, sys.stdout)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("objects_path")
    p.add_argument("relations_path")

    # p.add_argument("--freq_threshold", type=int, default=0)

    args = p.parse_args()
    main(args)
