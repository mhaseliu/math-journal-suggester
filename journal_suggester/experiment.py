"""Standalone preparation, training and evaluation for the published recipe."""
import argparse
from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import shutil
import subprocess
import sys

from .io import digest, journals, read_json, read_jsonl, request_digest, write_json, write_jsonl
from .records import identities
from .splits import quotas


def select_queries(references, weights, n, seed):
    counts = Counter(p['journal_id'] for p in references)
    if counts.keys() != weights.keys() or not 0 < n <= len(references):
        raise ValueError('Reference coverage or training size is invalid')
    allocation = quotas(n, weights, counts)
    selected, used = [], Counter()
    for p in sorted(references, key=lambda p: digest([seed, 'training', p['paper_id']])):
        if used[p['journal_id']] < allocation[p['journal_id']]:
            selected.append(p)
            used[p['journal_id']] += 1
    if len({p['group_id'] for p in selected}) != n:
        raise ValueError('Training requires distinct duplicate groups')
    return selected, allocation


def check_partitions(parts):
    """Training may be a reference subset; held-out identities must be separate."""
    names = ('reference', 'validation', 'test')
    for name in names:
        rows = parts[name]
        if not rows or len({p['paper_id'] for p in rows}) != len(rows) or len({p['group_id'] for p in rows}) != len(rows):
            raise ValueError(f'Empty or duplicate partition: {name}')
        if any(not p['title'].strip() or len(p['abstract'].split()) < 15 for p in rows):
            raise ValueError(f'Missing title or usable abstract: {name}')
    for a, b in combinations(names, 2):
        left, right = parts[a], parts[b]
        for keys in (lambda p: identities(p), lambda p: [p['group_id']],
                     lambda p: [digest(p['abstract'].casefold().split())]):
            if {k for p in left for k in keys(p)} & {k for p in right for k in keys(p)}:
                raise ValueError(f'Overlapping identities, groups or abstracts: {a}/{b}')


def select(split_dir, output, counts, config='configs/training-8000.json', models='configs/models.json'):
    source, root = Path(split_dir), Path(output)
    if root.exists():
        raise ValueError('Use a new experiment directory')
    spec, model_config = read_json(config), read_json(models)
    parts = {n: read_jsonl(source/f'{n}.jsonl') for n in ('reference', 'validation', 'test')}
    manifest = read_json(source/'manifest.json')
    for name, rows in parts.items():
        if digest(rows) != manifest['partition_hashes'][name]:
            raise ValueError(f'Frozen {name} partition was altered')
    check_partitions(parts)
    if len(parts['validation']) != spec['validation_papers']:
        raise ValueError('Validation size differs from training specification')
    parts['train'], allocation = select_queries(parts['reference'], read_json(counts), spec['training_papers'], spec['seed'])
    for name, rows in parts.items():
        write_jsonl(root/f'splits/{name}.jsonl', rows)
    write_json(root/'configs/models.json', model_config)
    write_json(root/'configs/training-8000.json', spec)
    hashes = {n: digest(v) for n, v in parts.items()}
    write_json(root/'splits/manifest.json', {'source_fingerprint': manifest['fingerprint'], 'partition_hashes': hashes})
    write_json(root/'selection.json', {'train_hash': hashes['train'], 'journal_quotas': allocation,
        'partition_hashes': hashes, 'models': model_config, 'config': spec, 'test_scored': False})
    return {'training_papers': len(parts['train']), 'validation_papers': len(parts['validation'])}


def load_selection(root):
    root = Path(root)
    saved = read_json(root/'selection.json')
    if saved['models'] != read_json(root/'configs/models.json') or saved['config'] != read_json(root/'configs/training-8000.json'):
        raise ValueError('Frozen experiment configuration changed')
    parts = {n: read_jsonl(root/f'splits/{n}.jsonl') for n in saved['partition_hashes']}
    if any(digest(v) != saved['partition_hashes'][n] for n,v in parts.items()):
        raise ValueError('Frozen experiment data changed')
    check_partitions(parts)
    return saved, parts


def prepare_partition(root, partition, parts, models, tokenizer):
    from .examples import build_request
    from .gpu import load_vectors
    from .input_audit import audit_inputs
    from .retrieval import shortlist
    from .web_ranker import CONTEXT
    import numpy as np
    root = Path(root)
    refs, queries = parts['reference'], parts[partition]
    rv = load_vectors(root/'vectors', 'reference', refs, models)
    qv = load_vectors(root/'vectors', partition, queries, models)
    names = {j['journal_id']: j['journal_name'] for j in journals()}
    requests, metadata = [], []
    training = partition == 'train'
    for i, query in enumerate(queries):
        candidates, info = shortlist(query, refs, (rv @ qv[i]).tolist(), top_k=20,
            target=query['journal_id'] if training else None)
        if len(candidates) != 20 or info['missing_evidence']:
            raise ValueError('Each request needs 20 journals and two distinct supporting papers per journal')
        request, lengths = build_request(query, candidates, names, tokenizer=tokenizer,
            label=query['journal_id'] if training else None, **CONTEXT, candidate_order='retrieval')
        if lengths['reference_limit'] != 100:
            raise ValueError('Evidence was shortened below the published 100-token allowance')
        requests.append(request)
        metadata.append({'paper_id': query['paper_id'], 'group_id': query['group_id'], 'target': query['journal_id'],
            'candidates': list(request['questions']['journal']['criteria']), 'request_hash': request_digest(request),
            'lengths': lengths, **info, 'evidence_ids': {c['journal_id']: [p['paper_id'] for p in c['references']] for c in candidates}})
    q = root/'qwen'
    write_jsonl(q/f'{partition}.jsonl', requests)
    write_jsonl(q/f'{partition}.meta.jsonl', metadata)
    audit = audit_inputs(root, root/'vectors', partition, queries, q, models, tokenizer, training=training)
    return requests, metadata, audit


def prepare(root, tokenizer=None):
    root = Path(root)
    if (root/'prepared.json').exists() or (root/'qwen').exists():
        raise ValueError('Prepared requests already exist; use a fresh output')
    saved, parts = load_selection(root)
    models, spec = saved['models'], saved['config']
    if tokenizer is None:
        from kev.model import load_tokenizer
        tokenizer = load_tokenizer(models['base_model'], models['base_revision'])
    train, tm, ta = prepare_partition(root, 'train', parts, models, tokenizer)
    val, vm, va = prepare_partition(root, 'validation', parts, models, tokenizer)
    test_ids = [{'paper_id': p['paper_id'], 'group_id': p['group_id']} for p in parts['test']]
    q = root/'qwen'
    write_jsonl(q/'test.meta.jsonl', test_ids)
    write_json(q/'report.json', {'backend': 'qwen3-embedding-8b', 'candidate_order': 'retrieval'})
    from .kev_runtime import validate_training
    validate_training(q, {**models, 'max_state': spec['max_state'], 'max_request': spec['max_request']}, tokenizer)
    prepared = {'config': spec, 'training_papers': len(train), 'train_request_hashes': [request_digest(r) for r in train],
        'train_metadata_hash': digest(tm), 'validation_request_hashes': [request_digest(r) for r in val],
        'validation_metadata_hash': digest(vm), 'test_identity_hash': digest(test_ids), 'journal_quotas': saved['journal_quotas'],
        'target_insertions': sum(m['inserted'] for m in tm), 'test_scored': False}
    write_json(root/'prepared.json', prepared)
    write_json(root/'local-audit.json', {'checks_pass': True, 'prepared_hash': digest(prepared),
        'train': ta, 'validation': va, 'no_prior_run_dependencies': True})
    longest = max(range(len(train)), key=lambda i: tm[i]['lengths']['request_tokens'])
    indices = [longest] + [i for i in range(len(train)) if i != longest][:23]
    pilot = root/'pilot-input'
    write_jsonl(pilot/'train.jsonl', [train[i] for i in indices])
    write_jsonl(pilot/'train.meta.jsonl', [tm[i] for i in indices])
    for name in ['validation.jsonl','validation.meta.jsonl','test.meta.jsonl','report.json']:
        shutil.copy2(q/name, pilot/name)
    return {'checks_pass': True, 'training_papers': len(train), 'target_insertions': prepared['target_insertions']}


def train(root, pilot=False):
    from .cloud_runtime import require_training_gpu
    require_training_gpu(36)
    from kev.checkpoint import Checkpoint
    from .kev_runtime import released, training_command
    root = Path(root).resolve()
    saved, _ = load_selection(root)
    models, spec = saved['models'], saved['config']
    prepared = read_json(root/'prepared.json')
    audit = read_json(root/'local-audit.json')
    if not audit['checks_pass'] or audit['prepared_hash'] != digest(prepared):
        raise ValueError('Preparation audit missing or changed')
    for part in ('train','validation'):
        if [request_digest(r) for r in read_jsonl(root/f'qwen/{part}.jsonl')] != prepared[f'{part}_request_hashes']:
            raise ValueError('Prepared requests changed')
        if digest(read_jsonl(root/f'qwen/{part}.meta.jsonl')) != prepared[f'{part}_metadata_hash']:
            raise ValueError('Prepared metadata changed')
    if not pilot and not read_json(root/'checkpoints/pilot/reload-check.json')['checks_pass']:
        raise ValueError('Run the three-update pilot first')
    ck = Checkpoint(released(models))
    directory = root/('pilot-input' if pilot else 'qwen')
    destination = root/'checkpoints'/('pilot' if pilot else 'full')
    command = training_command(directory, destination, ck, {**models,'max_state':5504,'max_request':6144},
        max_steps=0, learning_rate=spec['learning_rate'], preserve_candidate_order=True)[4:]
    command[command.index('--epochs')+1] = '1' if pilot else '5'
    wrapper = [sys.executable,'-u','-m','journal_suggester.b300_full_train','--journal-root',str(root)]
    if pilot:
        wrapper.append('--journal-pilot')
    subprocess.run(wrapper+command, cwd=root, check=True)
    return {'status': 'complete', 'checkpoint_directory': str(destination)}


def evaluate(root, checkpoint, partition='validation'):
    """One fixed comparison; no optimizer, tuning or choice of checkpoint here."""
    import gc
    import hashlib
    from .b300_precision import apply_precision
    from .b300_train import score_model
    from .evaluation import metrics
    from .kev_runtime import released
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import load_tokenizer
    root = Path(root)
    output = root/'evaluation'/partition
    if output.exists():
        raise ValueError('Existing evaluations are immutable; use a separate experiment')
    saved, parts = load_selection(root)
    model_config = saved['models']
    checkpoint = Path(checkpoint).resolve(strict=True)
    checkpoint_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()
        if p.name in {'adapter_model.safetensors','adapter_config.json','head.pt'}}
    if set(checkpoint_hashes) != {'adapter_model.safetensors','adapter_config.json','head.pt'}:
        raise ValueError('Incomplete selected checkpoint')
    write_json(output/'selection.json', {'checkpoint_files_sha256':checkpoint_hashes,
        'partition_hash':saved['partition_hashes'][partition], 'selection':'Fixed before this evaluation'})
    tok = load_tokenizer(model_config['base_model'],model_config['base_revision'])
    if partition == 'test':
        prepare_partition(root, partition, parts, model_config, tok)
    requests = read_jsonl(root/f'qwen/{partition}.jsonl')
    metadata = read_jsonl(root/f'qwen/{partition}.meta.jsonl')
    if len(metadata)!=len(requests) or any(m['inserted'] or 'label' in r['questions']['journal']
            or request_digest(r)!=m['request_hash'] for r,m in zip(requests,metadata)):
        raise ValueError('Evaluation requires frozen unlabelled natural candidates')
    predictions = [{'paper_id':m['paper_id'],'target':m['target'],'candidates':m['natural_journals'],
        'ranking':m['natural_journals'],'request_hash':m['request_hash']} for m in metadata]
    write_jsonl(output/'retrieval.predictions.jsonl',predictions)
    results = {'retrieval':metrics(predictions,20)}
    torch = apply_precision()
    for name,run in [('released_kev',released(model_config)),('fine_tuned_kev',str(checkpoint))]:
        tokenizer,model = Checkpoint(run).load('cuda',LoadOptions(dtype=torch.float32,merge=False,fused=False,cuda_graphs=False))
        rows = score_model(model,tokenizer,root/'qwen',output/name,partition=partition)
        results[name] = metrics(rows,20)
        del model,tokenizer
        gc.collect();torch.cuda.empty_cache()
    write_json(output/'results.json',{'metrics':results,'partition':partition,'optimizer_updates':0})
    return results


def verify_results(directory):
    """Recompute the public result table without a GPU, network or paper text."""
    from .evaluation import metrics
    directory = Path(directory)
    expected = read_json(directory/'test.json')['metrics']
    common = None
    for arm in ('retrieval','released_kev','fine_tuned_kev'):
        rows = read_jsonl(directory/f'predictions/{arm}.jsonl')
        identities = [(r['paper_id'],r['target'],r['candidates'],r['request_hash']) for r in rows]
        if len(rows)!=1000 or len({r['paper_id'] for r in rows})!=1000:
            raise ValueError('Expected 1,000 distinct test predictions')
        if common is not None and identities!=common:
            raise ValueError('Methods used different papers or candidates')
        common = identities
        for row in rows:
            if len(row['candidates'])!=20 or set(row['ranking'])!=set(row['candidates']) or len(row['ranking'])!=20:
                raise ValueError('Invalid candidate list or ranking')
            if 'probabilities' in row:
                probs = row['probabilities']
                if set(probs)!=set(row['candidates']) or not all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) or not math.isclose(sum(probs.values()),1,abs_tol=1e-5):
                    raise ValueError('Invalid saved probabilities')
                if row['ranking']!=sorted(probs,key=lambda k:(-probs[k],k)):
                    raise ValueError('Saved ranking disagrees with probabilities')
        actual = metrics(rows,20)
        for key in ('n','top1','top3','top5','candidate_recall_at_20'):
            if actual[key]!=expected[arm][key]:
                raise ValueError(f'Result mismatch: {arm}/{key}')
    return {'checks_pass':True,'test_papers':1000,'methods':3,'model_calls':0}


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; experiment integrity assertions must remain enabled")
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('select');q.add_argument('--splits',required=True);q.add_argument('--output',required=True)
    q.add_argument('--counts',default='data/publication-counts.json');q.add_argument('--config',default='configs/training-8000.json')
    q=sub.add_parser('embed');q.add_argument('--root',required=True);q.add_argument('--partitions',nargs='+',default=['reference','train','validation'],choices=['reference','train','validation','test'])
    q=sub.add_parser('prepare');q.add_argument('--root',required=True)
    q=sub.add_parser('train');q.add_argument('--root',required=True);q.add_argument('--pilot',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--root',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--partition',choices=['validation','test'],default='validation')
    q=sub.add_parser('verify-results');q.add_argument('--directory',default='results')
    a=p.parse_args()
    if a.command=='select':result=select(a.splits,a.output,a.counts,a.config)
    elif a.command=='embed':
        from .gpu import embed_split
        saved,_=load_selection(a.root)
        result=embed_split(Path(a.root)/'splits',Path(a.root)/'vectors',saved['models'],a.partitions)
    elif a.command=='prepare':result=prepare(a.root)
    elif a.command=='train':result=train(a.root,a.pilot)
    elif a.command=='evaluate':result=evaluate(a.root,a.checkpoint,a.partition)
    else:result=verify_results(a.directory)
    if result is not None:print(__import__('json').dumps(result,indent=2))

if __name__=='__main__':
    main()
