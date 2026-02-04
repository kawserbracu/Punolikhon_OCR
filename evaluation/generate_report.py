import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description='Generate a short report combining recognition and detection metrics')
    ap.add_argument('--rec_eval', type=str, required=True, help='Path to recognition_eval.json')
    ap.add_argument('--det_eval', type=str, nargs='*', default=[], help='Pairs of label=path_to_detection_eval.json (supports multiple)')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = load_json(Path(args.rec_eval))

    # Parse det_eval as label=path pairs
    dets = {}
    for entry in args.det_eval:
        if '=' in entry:
            label, path = entry.split('=', 1)
            dets[label] = load_json(Path(path))

    # Write markdown
    md = [
        '# OCR Report',
        '',
        '## Recognition Metrics',
        '',
    ]

    if rec:
        for k, v in rec.items():
            md.append(f'- **{k}**: {v}')
    else:
        md.append('- **recognition**: not found')

    md += ['', '## Detection Metrics', '']
    if dets:
        for label, det in dets.items():
            md.append(f'### {label}')
            if det:
                for k, v in det.items():
                    md.append(f'- **{k}**: {v}')
            else:
                md.append('- no data')
            md.append('')
    else:
        md.append('- no detection evaluations provided')

    (out_dir / 'report.md').write_text('\n'.join(md), encoding='utf-8')

    # Also dump a JSON summary
    summary = {'recognition': rec, 'detection': dets}
    (out_dir / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Saved report to', out_dir)


if __name__ == '__main__':
    main()
