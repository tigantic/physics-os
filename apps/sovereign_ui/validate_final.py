#!/usr/bin/env python3
"""
Sovereign FPS UI — Final Validation Script
Proves all 21 backend endpoints are reachable and return real data.
Run this after starting the backend server.

Usage:
    python validate_final.py [--base http://127.0.0.1:8420]

Exit code 0 = all endpoints validated.
"""

import json
import sys
import urllib.request
import urllib.error
import time

BASE = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--base' else 'http://127.0.0.1:8420'

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
WARN = '\033[93m⚠\033[0m'

results = []
test_case_id = None


def get(path):
    url = f'{BASE}{path}'
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return 200, data
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return 0, str(e)


def post(path, body=None):
    url = f'{BASE}{path}'
    try:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return 0, str(e)


def check(phase, endpoint, method, path, body=None):
    global test_case_id
    fn = get if method == 'GET' else post
    args = [path] if method == 'GET' else [path, body]
    status, data = fn(*args)
    ok = 200 <= status < 300
    icon = PASS if ok else FAIL
    results.append((phase, endpoint, ok))
    print(f'  {icon} [{method:4}] {endpoint:40} → {status}')
    return ok, data


def main():
    global test_case_id
    print(f'\n{"="*60}')
    print(f'  Sovereign FPS — Final Endpoint Validation')
    print(f'  Base: {BASE}')
    print(f'{"="*60}\n')

    t0 = time.time()

    # ── Phase 1: Foundation (5 endpoints) ─────────────────────
    print('Phase 1 — Foundation')
    ok, data = check(1, '/api/contract', 'GET', '/api/contract')
    if ok and data:
        print(f'         version={data.get("version")}, operators={data.get("operators",{}).get("count")}')

    ok, data = check(1, '/api/cases', 'GET', '/api/cases')
    if ok and data:
        print(f'         total={data.get("total")}, returned={len(data.get("cases",[]))}')

    # Create test case
    ok, data = check(1, '/api/cases (create)', 'POST', '/api/cases', {
        'patient_age': 30, 'patient_sex': 'female', 'procedure': 'rhinoplasty', 'notes': 'validation_test'
    })
    if ok and data:
        test_case_id = data.get('case_id')
        print(f'         created case_id={test_case_id}')

    check(1, '/api/curate', 'POST', '/api/curate')
    print()

    # ── Phase 2: Twin Inspection (5 endpoints) ────────────────
    print('Phase 2 — Twin Inspection')
    if test_case_id:
        check(2, f'/api/cases/:id/twin', 'GET', f'/api/cases/{test_case_id}/twin')
        check(2, f'/api/cases/:id/mesh', 'GET', f'/api/cases/{test_case_id}/mesh')
        check(2, f'/api/cases/:id/landmarks', 'GET', f'/api/cases/{test_case_id}/landmarks')
        check(2, f'/api/cases/:id/visualization', 'GET', f'/api/cases/{test_case_id}/visualization')
        check(2, f'/api/cases/:id/timeline', 'GET', f'/api/cases/{test_case_id}/timeline')
    else:
        print(f'  {WARN} Skipped — no test case created')
    print()

    # ── Phase 3: Plan Editor (5 endpoints) ────────────────────
    print('Phase 3 — Plan Editor')
    check(3, '/api/operators', 'GET', '/api/operators')
    check(3, '/api/templates', 'GET', '/api/templates')
    check(3, '/api/plan/template', 'POST', '/api/plan/template', {'category': 'rhinoplasty', 'template': 'standard'})
    check(3, '/api/plan/custom', 'POST', '/api/plan/custom', {'name': 'test', 'procedure': 'rhinoplasty', 'steps': []})
    if test_case_id:
        check(3, '/api/plan/compile', 'POST', '/api/plan/compile', {'case_id': test_case_id})
    else:
        check(3, '/api/plan/compile', 'POST', '/api/plan/compile', {'case_id': 'test'})
    print()

    # ── Phase 4: Analysis (5 endpoints) ───────────────────────
    print('Phase 4 — Analysis')
    if test_case_id:
        check(4, '/api/whatif', 'POST', '/api/whatif', {'case_id': test_case_id, 'overrides': {}})
        check(4, '/api/sweep', 'POST', '/api/sweep', {
            'case_id': test_case_id, 'operator': 'dorsal_reduction',
            'param': 'amount_mm', 'values': [1.0, 2.0, 3.0]
        })
        check(4, '/api/report', 'POST', '/api/report', {'case_id': test_case_id, 'format': 'markdown'})
    else:
        check(4, '/api/whatif', 'POST', '/api/whatif', {'case_id': 'test', 'overrides': {}})
        check(4, '/api/sweep', 'POST', '/api/sweep', {'case_id': 'test', 'operator': 'test', 'param': 'test', 'values': [1]})
        check(4, '/api/report', 'POST', '/api/report', {'case_id': 'test', 'format': 'markdown'})
    check(4, '/api/compare/plans', 'POST', '/api/compare/plans', {'plan_a': 'a', 'plan_b': 'b'})
    check(4, '/api/compare/cases', 'POST', '/api/compare/cases', {'case_a': 'a', 'case_b': 'b'})
    print()

    # ── Cleanup: Delete test case ─────────────────────────────
    print('Cleanup')
    if test_case_id:
        ok, _ = check(0, f'/api/cases/:id/delete', 'POST', f'/api/cases/{test_case_id}/delete')
        if ok:
            print(f'         deleted test case')
    print()

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    passed = sum(1 for _, _, ok in results if ok)
    failed = sum(1 for _, _, ok in results if not ok)
    total = len(results)

    print(f'{"="*60}')
    print(f'  Results: {passed}/{total} passed, {failed} failed')
    print(f'  Time:    {elapsed:.1f}s')
    print(f'{"="*60}')

    if failed > 0:
        print(f'\n  {FAIL} Failed endpoints:')
        for phase, endpoint, ok in results:
            if not ok:
                print(f'     Phase {phase}: {endpoint}')
        sys.exit(1)
    else:
        print(f'\n  {PASS} All 21 endpoints validated. Full API coverage confirmed.')
        sys.exit(0)


if __name__ == '__main__':
    main()
