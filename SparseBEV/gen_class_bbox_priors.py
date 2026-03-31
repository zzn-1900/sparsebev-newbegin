import argparse
import pickle
from collections import defaultdict

try:
    import mmcv
except ImportError:
    mmcv = None


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError('This script requires numpy to compute statistics.') from exc
    return np


DEFAULT_CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate per-class bbox priors from a nuScenes info pickle.')
    parser.add_argument(
        '--ann-file',
        default='data/nuscenes/nuscenes_infos_train_sweep.pkl',
        help='Path to the nuScenes info pickle.')
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=DEFAULT_CLASS_NAMES,
        help='Class order used when printing config-ready priors.')
    parser.add_argument(
        '--stat',
        choices=['median', 'mean'],
        default='median',
        help='Statistic used for the exported config priors.')
    parser.add_argument(
        '--precision',
        type=int,
        default=2,
        help='Decimal precision used when printing config priors.')
    return parser.parse_args()


def load_infos(path):
    if mmcv is not None:
        try:
            data = mmcv.load(path)
        except Exception:
            with open(path, 'rb') as f:
                data = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    elif isinstance(data, list):
        infos = data
    else:
        raise ValueError('Unsupported annotation structure in {}.'.format(path))

    if not infos:
        raise ValueError('No infos found in {}.'.format(path))

    return infos


def _to_numpy_boxes(boxes):
    np = require_numpy()
    if isinstance(boxes, np.ndarray):
        return boxes
    if hasattr(boxes, 'tensor'):
        tensor = boxes.tensor
        return tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.asarray(tensor)
    return np.asarray(boxes)


def extract_boxes_and_names(info):
    np = require_numpy()
    box_keys = ['gt_boxes', 'gt_boxes_3d', 'gt_bboxes_3d']
    name_keys = ['gt_names', 'gt_names_3d']

    boxes = None
    names = None

    for key in box_keys:
        if key in info:
            boxes = _to_numpy_boxes(info[key])
            break

    for key in name_keys:
        if key in info:
            names = np.asarray(info[key])
            break

    if boxes is None or names is None:
        raise KeyError(
            'Failed to find gt boxes / gt names. Available keys: {}'.format(sorted(info.keys())))

    if boxes.shape[0] != len(names):
        raise ValueError(
            'Mismatched gt_boxes ({}) and gt_names ({}) lengths.'.format(boxes.shape[0], len(names)))

    valid_flag = info.get('valid_flag')
    if valid_flag is not None:
        valid_flag = np.asarray(valid_flag).astype(bool)
        boxes = boxes[valid_flag]
        names = names[valid_flag]

    return boxes, names


def summarize(values):
    np = require_numpy()
    arr = np.asarray(values, dtype=np.float32)
    return {
        'count': int(arr.shape[0]),
        'median': np.median(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0),
    }


def format_prior_line(name, stats, stat_key, precision):
    z, w, l, h = stats[stat_key]
    fmt = '{{:.{}f}}'.format(precision)
    return (
        '    dict(z={z}, w={w}, l={l}, h={h}),   # {name}, count={count}'.format(
            z=fmt.format(float(z)),
            w=fmt.format(float(w)),
            l=fmt.format(float(l)),
            h=fmt.format(float(h)),
            name=name,
            count=stats['count'],
        )
    )


def main():
    args = parse_args()
    infos = load_infos(args.ann_file)
    np = require_numpy()

    class_values = defaultdict(list)
    unknown_names = set()

    for info in infos:
        boxes, names = extract_boxes_and_names(info)
        for box, name in zip(boxes, names):
            name = str(name)
            if name not in args.class_names:
                unknown_names.add(name)
                continue
            if box.shape[0] < 6:
                raise ValueError('Each gt box must contain at least 6 values, got {}.'.format(box.shape[0]))

            z = float(box[2])
            w = float(box[3])
            l = float(box[4])
            h = float(box[5])
            class_values[name].append([z, w, l, h])

    print('Loaded {} infos from {}'.format(len(infos), args.ann_file))
    if unknown_names:
        print('Ignored classes not in --class-names: {}'.format(sorted(unknown_names)))
    print('')

    per_class_stats = {}
    print('Per-class statistics:')
    for name in args.class_names:
        values = class_values.get(name, [])
        if not values:
            print('  {}: no samples'.format(name))
            continue

        stats = summarize(values)
        per_class_stats[name] = stats
        z_med, w_med, l_med, h_med = stats['median']
        z_mean, w_mean, l_mean, h_mean = stats['mean']
        print(
            '  {name}: count={count}, median(z/w/l/h)=({mz:.3f}, {mw:.3f}, {ml:.3f}, {mh:.3f}), '
            'mean=({ez:.3f}, {ew:.3f}, {el:.3f}, {eh:.3f})'.format(
                name=name,
                count=stats['count'],
                mz=float(z_med),
                mw=float(w_med),
                ml=float(l_med),
                mh=float(h_med),
                ez=float(z_mean),
                ew=float(w_mean),
                el=float(l_mean),
                eh=float(h_mean),
            )
        )

    missing = [name for name in args.class_names if name not in per_class_stats]
    if missing:
        raise ValueError('Missing classes in annotations: {}'.format(missing))

    print('')
    print('query_class_bbox_priors = [')
    for name in args.class_names:
        print(format_prior_line(name, per_class_stats[name], args.stat, args.precision))
    print(']')


if __name__ == '__main__':
    main()
