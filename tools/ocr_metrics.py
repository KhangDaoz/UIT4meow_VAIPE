"""OCR metrics for block C (plan.md 5.3 / 6): CER, WER, exact-match accuracy, overall and per class.

Before scoring, both strings are normalised: Unicode NFC, whitespace collapsed to one space.
Case is kept (upper/lower case are different characters).

    CER = sum(edit distance in characters) / sum(characters in the ground truth)   (corpus CER)
    WER = same with words
    acc = share of lines read exactly right

Usage as a library:
    from ocr_metrics import evaluate
    evaluate(truths, preds, labels)   # labels: class of each line, or None
"""
import unicodedata


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', str(s)).split())


def edit_distance(a, b):
    """Levenshtein distance between two sequences (strings or lists of words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _score(pairs):
    char_err = char_tot = word_err = word_tot = exact = 0
    for t, p in pairs:
        char_err += edit_distance(t, p)
        char_tot += len(t)
        tw, pw = t.split(), p.split()
        word_err += edit_distance(tw, pw)
        word_tot += len(tw)
        exact += t == p
    n = len(pairs)
    return {
        'lines': n,
        'cer': char_err / max(char_tot, 1),
        'wer': word_err / max(word_tot, 1),
        'acc_full_sequence': exact / max(n, 1),
    }


def evaluate(truths, preds, labels=None):
    """{'all': {...}, 'per_class': {label: {...}}} with lines, cer, wer, acc_full_sequence."""
    assert len(truths) == len(preds)
    pairs = [(norm(t), norm(p)) for t, p in zip(truths, preds)]
    result = {'all': _score(pairs)}
    if labels is not None:
        groups = {}
        for lab, pair in zip(labels, pairs):
            groups.setdefault(lab, []).append(pair)
        result['per_class'] = {lab: _score(g) for lab, g in sorted(groups.items())}
    return result


def format_table(result):
    rows = [('all', result['all'])] + list(result.get('per_class', {}).items())
    out = [f'{"class":10s} {"lines":>6s} {"CER":>7s} {"WER":>7s} {"acc":>7s}']
    for name, r in rows:
        out.append(f'{name:10s} {r["lines"]:6d} {r["cer"]:7.4f} {r["wer"]:7.4f} '
                   f'{r["acc_full_sequence"]:7.4f}')
    return '\n'.join(out)
