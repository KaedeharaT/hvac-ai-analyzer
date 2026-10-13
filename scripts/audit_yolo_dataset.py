"""Read-only audit of an explicitly selected local YOLO dataset.

Writes only the requested JSON report. Never changes labels, split or images.
Reports may contain private filenames and must remain outside Git.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def audit(root: Path, class_count: int = 3) -> dict:
    if not root.is_dir():
        raise ValueError('Dataset directory does not exist.')
    report = {'splits': {}, 'invalid_labels': [], 'missing_labels': [], 'orphan_labels': []}
    hashes, families = defaultdict(list), defaultdict(set)
    for split in ('train', 'valid', 'val', 'test'):
        directory = root / split
        if not directory.exists():
            continue
        images = sorted(p for p in (directory / 'images').glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        labels = sorted((directory / 'labels').glob('*.txt'))
        classes, labelled_images, empty = Counter(), Counter(), 0
        for label in labels:
            seen = set()
            lines = label.read_text(encoding='utf-8-sig').splitlines()
            empty += not bool(lines)
            for number, line in enumerate(lines, 1):
                try:
                    c, x, y, w, h = map(float, line.split())
                    if not (all(math.isfinite(v) for v in (c,x,y,w,h))
                            and c.is_integer() and 0 <= c < class_count
                            and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                            and x-w/2 >= -1e-5 and y-h/2 >= -1e-5
                            and x+w/2 <= 1+1e-5 and y+h/2 <= 1+1e-5):
                        raise ValueError('Invalid normalized box')
                    classes[int(c)] += 1; seen.add(int(c))
                except ValueError:
                    report['invalid_labels'].append({'file': str(label.relative_to(root)), 'line': number})
            labelled_images.update(seen)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        report['missing_labels'] += [str(p.relative_to(root)) for p in images if p.stem not in label_stems]
        report['orphan_labels'] += [str(p.relative_to(root)) for p in labels if p.stem not in image_stems]
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            hashes[digest].append({'split': split, 'file': str(image.relative_to(root))})
            # Roboflow siblings can have different encoded pixels but the same source stem.
            families[image.stem.split('.rf.')[0]].add(split)
        report['splits'][split] = {'images': len(images), 'labels': len(labels), 'empty_labels': empty,
                                   'boxes_by_class': dict(classes), 'images_by_class': dict(labelled_images)}
    report['exact_cross_split_duplicates'] = [rows for rows in hashes.values() if len({r['split'] for r in rows}) > 1]
    report['source_name_cross_split_candidates'] = {k: sorted(v) for k,v in families.items() if len(v)>1}
    report['unique_image_hashes'] = len(hashes)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('Write audit reports outside the source dataset.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(args.dataset), ensure_ascii=False, indent=2), encoding='utf-8')
