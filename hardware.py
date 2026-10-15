"""Run the CSP Grover circuits on real IBM Quantum hardware (or a fake backend for a dry run).

One-time setup (free "Open" plan, about 10 QPU-minutes a month):
    1. Create an account at https://quantum.cloud.ibm.com and copy your API key
       and instance CRN.
    2. python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
S.save_account(channel='ibm_quantum_platform', token='<API KEY>', instance='<CRN>', set_as_default=True)"

Then:
    python hardware.py --dry-run                 # full pipeline on FakeTorino, no account needed
    python hardware.py --list                    # backends available to you
    python hardware.py --backend ibm_torino      # submit (asks for confirmation first)
    python hardware.py --collect <JOB_ID>        # fetch a finished job later

Results go to results/hardware_<backend>_<job>.json and results/hardware.csv.
The circuits are the seed-0 instances of study.py, both oracle designs.
Each instance is run at the optimal iteration count and at 0 iterations
(a calibration point: plain random sampling, which isolates readout error).
"""

import argparse
import csv
import datetime as dt
import json
import os

import csp
import study

OUT = "results"
DEFAULT_SHOTS = 4096


def selected_instances(max_qubits):
    picks = [x for x in study.instance_suite() if x[2] == 0 and x[3].n_qubits <= max_qubits]
    return picks


def build_jobs(backend, max_qubits, synth_modes):
    jobs = []
    for family, label, seed, p in selected_instances(max_qubits):
        M = len(p.solutions())
        k_opt = csp.optimal_iterations(p.N, M)
        for syn in synth_modes:
            for k in sorted({0, k_opt}):
                tqc = csp.transpile_grover(p, k, synthesis=syn, backend=backend, optimization_level=3)
                jobs.append({"family": family, "size": label, "problem": p.name, "synthesis": syn,
                             "iterations": k, "optimal_iterations": k_opt, "N": p.N, "M": M,
                             "two_qubit_gates": csp.circuit_stats(tqc)["cx"], "depth": tqc.depth(),
                             "circuit": tqc, "problem_obj": p})
    return jobs


def summarize(jobs, counts_list, backend_name, job_id, shots):
    rows = []
    for j, counts in zip(jobs, counts_list):
        p = j["problem_obj"]
        succ = csp.success_from_counts(p, counts)
        hits = round(succ * shots)
        lo, hi = csp.wilson_interval(hits, shots)
        rows.append({
            "backend": backend_name, "job_id": job_id,
            "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **{k: j[k] for k in ("family", "size", "problem", "synthesis", "iterations",
                                 "optimal_iterations", "N", "M", "two_qubit_gates", "depth")},
            "shots": shots, "hits": hits, "success": succ, "ci_low": lo, "ci_high": hi,
            "ideal": csp.ideal_success_probability(p.N, j["M"], j["iterations"]),
            "random_guess": j["M"] / j["N"],
            "counts": json.dumps(counts),
        })
    return rows


def save(rows, backend_name, job_id):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/hardware_{backend_name}_{job_id}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    path = f"{OUT}/hardware.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "counts"], extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['problem']:24s} {r['synthesis']:8s} k={r['iterations']} 2q={r['two_qubit_gates']:5d} "
              f"success={r['success']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
              f"ideal={r['ideal']:.3f} random={r['random_guess']:.3f}")


def counts_from_result(result):
    # The measurement register in build_grover_circuit is named "m".
    return [pub.data.m.get_counts() for pub in result]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", help="IBM backend name, e.g. ibm_torino")
    ap.add_argument("--dry-run", action="store_true", help="run on FakeTorino locally")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", metavar="JOB_ID")
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--max-qubits", type=int, default=11,
                    help="skip instances with more logical qubits (deep circuits only return noise)")
    ap.add_argument("--synthesis", nargs="+", default=["baseline", "aux+rp"], choices=csp.SYNTHESIS_MODES)
    ap.add_argument("--yes", action="store_true", help="do not ask before submitting")
    args = ap.parse_args()

    if args.dry_run:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
        jobs = build_jobs(backend, args.max_qubits, args.synthesis)
        counts_list = []
        for j in jobs:
            small, nm = csp.compact_device_circuit(j["circuit"], backend)
            r = AerSimulator(noise_model=nm, seed_simulator=5).run(small, shots=args.shots).result()
            counts_list.append({k.replace(" ", ""): v for k, v in r.get_counts().items()})
        save(summarize(jobs, counts_list, "dryrun_fake_torino", "local", args.shots), "dryrun_fake_torino", "local")
        return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    if args.list:
        for b in service.backends(operational=True, simulator=False):
            st = b.status()
            print(f"{b.name:16s} {b.num_qubits:4d} qubits  pending jobs: {st.pending_jobs}")
        return
    if args.collect:
        # Circuits are rebuilt in the same order as at submission (fixed transpiler seed);
        # use the same --max-qubits / --synthesis / --shots you submitted with.
        job = service.job(args.collect)
        backend = job.backend()
        jobs = build_jobs(backend, args.max_qubits, args.synthesis)
        save(summarize(jobs, counts_from_result(job.result()), backend.name, job.job_id(), args.shots),
             backend.name, job.job_id())
        return

    backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False)
    jobs = build_jobs(backend, args.max_qubits, args.synthesis)
    total_2q = sum(j["two_qubit_gates"] for j in jobs)
    print(f"Backend {backend.name}: {len(jobs)} circuits x {args.shots} shots, {total_2q} two-qubit gates in total.")
    for j in jobs:
        print(f"  {j['problem']:24s} {j['synthesis']:8s} k={j['iterations']} 2q={j['two_qubit_gates']}")
    if not args.yes and input("Submit to real hardware? This uses your QPU time. [y/N] ").strip().lower() != "y":
        print("Not submitted.")
        return
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = args.shots
    job = sampler.run([j["circuit"] for j in jobs])
    print(f"Submitted job {job.job_id()}. Waiting (you can Ctrl+C and later run --collect {job.job_id()}) ...")
    result = job.result()
    save(summarize(jobs, counts_from_result(result), backend.name, job.job_id(), args.shots),
         backend.name, job.job_id())


if __name__ == "__main__":
    main()
