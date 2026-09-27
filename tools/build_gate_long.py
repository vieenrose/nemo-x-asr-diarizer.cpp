#!/usr/bin/env python3
"""Build gate_long: a >= 5 min, 6-speaker, zh-TW/en conversation for end-to-end evaluation.

What it adds over the 45-65 s gates: length (speaker cache and streaming state have to hold up for minutes,
not seconds), recurring speakers (a voice must be recognised again after several other turns), frequent
language switches, and a few short overlaps.

Sources (both local, both openly licensed, real speech only - no synthesis, no splicing inside a turn):
  * zh-TW: Mozilla Common Voice 17.0 zh-TW test split (CC0), up_votes >= 2 and down_votes == 0
  * en:    LibriSpeech test-clean (CC BY 4.0)
Texts already used by another gate manifest are excluded, so nothing overlaps an existing gate.

Deterministic: fixed seed, selection by metadata only, never by model output. Writes, into eval-bilingual/:
  gate_long.wav (24 kHz mono PCM16, same as the other gates), manifest_long.json (score_stream.py format),
  gate_long.ref.txt, turns_long.tsv (score_turns.py format).
"""
import csv, glob, hashlib, json, os, random, subprocess, sys, wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EV = os.path.join(ROOT, 'eval-bilingual')
RAW = os.path.join(EV, 'raw')
LIBRI = os.environ.get('LIBRISPEECH', '/home/user/dl/LibriSpeech/test-clean')
SR, SEED = 24000, 2026
TARGET_S = 330.0              # stop adding turns once past this (>= 5.5 min)
ZH_VOICES, EN_VOICES = 4, 2
OVERLAPS = 4                  # turns that start before the previous one ends
DUR = {"zh-TW": (2.0, 12.0), "en": (3.0, 12.0)}   # per-turn duration bounds, seconds


def decode(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'],
                       capture_output=True, check=True)
    return p.stdout


def used_texts():
    out = set()
    for m in glob.glob(os.path.join(EV, 'manifest_*.json')):
        if m.endswith('manifest_long.json'):
            continue
        for t in json.load(open(m, encoding='utf-8')).get('table', []):
            out.add(t.get('text', '').strip().lower())
    return out


def main():
    rng = random.Random(SEED)
    used = used_texts()

    # zh-TW voices with enough qualifying clips
    clip_index = {os.path.basename(p): p for p in glob.glob(os.path.join(RAW, 'clips', '**', '*.mp3'), recursive=True)}
    rows = list(csv.DictReader(open(os.path.join(RAW, 'cv17_zhtw_test.tsv'), encoding='utf-8'), delimiter='\t'))
    by_voice = {}
    for r in rows:
        if int(r['up_votes'] or 0) >= 2 and int(r['down_votes'] or 0) == 0 and r['path'] in clip_index \
                and len(r['sentence']) >= 6 and r['sentence'].strip().lower() not in used:
            by_voice.setdefault(r['client_id'], []).append(r)
    zh_voices = sorted(v for v, rs in by_voice.items() if len(rs) >= 9)
    zh_pick = rng.sample(zh_voices, ZH_VOICES)

    # LibriSpeech speakers
    spk_dirs = sorted(d for d in os.listdir(LIBRI) if d.isdigit())
    en_pick = rng.sample(spk_dirs, EN_VOICES)
    en_utts = {}
    for s in en_pick:
        utts = []
        for tr in sorted(glob.glob(os.path.join(LIBRI, s, '*', '*.trans.txt'))):
            for line in open(tr):
                uid, text = line.strip().split(' ', 1)
                if text.lower() not in used:
                    utts.append((os.path.join(os.path.dirname(tr), uid + '.flac'), text))
        rng.shuffle(utts)
        en_utts[s] = utts

    roles = [{'gold': f'S{i+1}', 'lang': 'zh-TW', 'voice': 'commonvoice-zhTW-' + v[:8], 'src_voice': v}
             for i, v in enumerate(zh_pick)]
    roles += [{'gold': f'S{ZH_VOICES+i+1}', 'lang': 'en', 'voice': f'librispeech-{s}', 'src_voice': s}
              for i, s in enumerate(en_pick)]
    queues = {r['gold']: (rng.sample(by_voice[r['src_voice']], len(by_voice[r['src_voice']]))
                          if r['lang'] == 'zh-TW' else list(en_utts[r['src_voice']])) for r in roles}

    audio, table, t_end, prev_gold = [], [], 0.0, None
    overlap_turns = set(rng.sample(range(4, 40), OVERLAPS))
    turn = 0
    while t_end < TARGET_S:
        # every voice recurs; never the same voice twice in a row
        cands = [r for r in roles if r['gold'] != prev_gold and queues[r['gold']]]
        # balance the two languages by accumulated speech time (zh-TW clips are shorter than LibriSpeech's)
        spoken = {l: sum(t['dur_s'] for t in table if t['lang'] == l) for l in ('zh-TW', 'en')}
        behind = min(spoken, key=spoken.get)
        lagging = [r for r in cands if r['lang'] == behind]
        pool = lagging if (lagging and rng.random() < 0.8) else cands
        role = rng.choice(pool)
        q = queues[role['gold']]
        while q:
            item = q.pop()
            if role['lang'] == 'zh-TW':
                src, text = clip_index[item['path']], item['sentence']
            else:
                src, text = item
            pcm = decode(src)
            dur = len(pcm) / 2 / SR
            if DUR[role["lang"]][0] <= dur <= DUR[role["lang"]][1]:
                break
        else:
            continue
        turn += 1
        if turn in overlap_turns and table:
            start = max(table[-1]['start_s'] + 1.0, t_end - rng.uniform(0.5, 1.2))
        else:
            start = t_end + rng.uniform(0.15, 0.9)
        table.append({'turn': turn, 'gold': role['gold'], 'lang': role['lang'], 'voice': role['voice'],
                      'start_s': round(start, 3), 'end_s': round(start + dur, 3), 'dur_s': round(dur, 3),
                      'src': os.path.relpath(src, ROOT), 'sha': hashlib.sha256(pcm).hexdigest()[:16],
                      'text': text, 'overlap': start < t_end})
        audio.append((start, pcm))
        t_end = max(t_end, start + dur)
        prev_gold = role['gold']

    n = int(round((t_end + 0.5) * SR))
    mix = [0] * n
    import array
    for start, pcm in audio:
        a = array.array('h', pcm)
        o = int(round(start * SR))
        for i, v in enumerate(a):
            mix[o + i] += v
    out = array.array('h', (max(-32768, min(32767, v)) for v in mix)).tobytes()
    wav_path = os.path.join(EV, 'gate_long.wav')
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out)

    ref = ' '.join(t['text'] for t in sorted(table, key=lambda t: t['start_s']))
    open(os.path.join(EV, 'gate_long.ref.txt'), 'w', encoding='utf-8').write(ref + '\n')
    with open(os.path.join(EV, 'turns_long.tsv'), 'w', encoding='utf-8') as f:
        f.write('turn\tspeaker\tlang\tvoice\tstart_s\tend_s\tdur_s\tclip_sha256\ttext\n')
        for t in table:
            f.write(f"{t['turn']}\t{t['gold']}\t{t['lang']}\t{t['voice']}\t{t['start_s']}\t{t['end_s']}\t"
                    f"{t['dur_s']}\t{t['sha']}\t{t['text']}\n")
    man = {'builder': 'nemo-x-asr-diarizer.cpp/tools/build_gate_long.py', 'seed': SEED,
           'window_samples': 83200, 'hop_samples': 70400,
           'design': f'{len(roles)} recurring voices, alternating zh-TW/en turns, {OVERLAPS} short overlaps',
           'sources': {'zh-TW': 'Mozilla Common Voice 17.0 zh-TW test split (CC0), up_votes>=2 & down_votes==0',
                       'en': 'LibriSpeech test-clean (CC BY 4.0)'},
           'gold_speakers': len(roles), 'turns': len(table), 'total_s': round(n / SR, 3),
           'wav_sha256': hashlib.sha256(open(wav_path, 'rb').read()).hexdigest(),
           'roles': [{k: r[k] for k in ('gold', 'lang', 'voice')} for r in roles], 'table': table}
    json.dump(man, open(os.path.join(EV, 'manifest_long.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    zh = sum(t['dur_s'] for t in table if t['lang'] == 'zh-TW'); en = sum(t['dur_s'] for t in table if t['lang'] == 'en')
    print(f"gate_long.wav: {n/SR:.1f} s, {len(table)} turns, {len(roles)} voices, zh {zh:.0f} s / en {en:.0f} s, "
          f"{sum(t['overlap'] for t in table)} overlaps")


if __name__ == '__main__':
    main()
