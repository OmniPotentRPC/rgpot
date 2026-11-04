configfile: "benchflow/config.yaml"


rule all:
    input:
        "benchflow/out/timing.json",
        "benchflow/out/memory.json",
        "benchflow/out/noise.json",
        "benchflow/out/thresholds.json",


rule machine:
    output:
        "benchflow/out/home/.asv-machine.json",
        touch("benchflow/out/machine.done"),
    params:
        machine=config["machine"],
    shell:
        """
        set -euo pipefail
        . benchflow/env.sh
        python benchflow/export_results.py prepare-machine
        asv machine --yes --machine {params.machine} \
            --os Linux --arch x86_64 --cpu unspecified --num_cpu 1 --ram unspecified
        python benchflow/export_results.py check-machine
        """


rule time_run:
    input:
        "benchflow/out/machine.done",
    output:
        touch("benchflow/out/time.done"),
    params:
        rounds=config["rounds"],
        runs=config["runs"],
        machine=config["machine"],
    shell:
        """
        set -euo pipefail
        . benchflow/env.sh
        mkdir -p benchflow/out/runs benchflow/out/results
        rm -f benchflow/out/runs/*.json
        main=$(git rev-parse develop)
        printf '%s\n' "$main" > benchflow/out/commits.txt
        i=1
        while [ "$i" -le {params.runs} ]; do
          id=$(printf '%02d' "$i")
          for side in a b; do
            conf=".asv-run-time-$id-$side.conf.json"
            results="benchflow/out/results/time-$id-$side"
            rm -rf "$results"
            python benchflow/export_results.py derive-conf asv.conf.json "$conf" "$results"
            asv run "$main^!" --interleave-rounds --record-samples \
              -a rounds={params.rounds} --bench time_ --machine {params.machine} \
              --config "$conf" --show-stderr
          done
          python benchflow/export_results.py save-pair "$i" \
            --earlier "benchflow/out/results/time-$id-a" \
            --later "benchflow/out/results/time-$id-b" \
            --commit "$main" --machine {params.machine}
          i=$((i + 1))
        done
        """


rule export_time:
    input:
        "benchflow/out/time.done",
    output:
        "benchflow/out/timing.json",
        "benchflow/out/noise.json",
        "benchflow/out/thresholds.json",
    shell:
        "python benchflow/export_results.py aggregate"


rule memory_run:
    input:
        "benchflow/out/timing.json",
        "benchflow/out/noise.json",
        "benchflow/out/thresholds.json",
    output:
        "benchflow/out/memory.json",
    params:
        head=config["head"],
        machine=config["machine"],
    shell:
        """
        set -euo pipefail
        . benchflow/env.sh
        head=$(git rev-parse {params.head})
        for i in 1 2; do
          id=$(printf '%02d' "$i")
          conf=".asv-run-memory-$id.conf.json"
          results="benchflow/out/results/memory-$id"
          rm -rf "$results"
          python benchflow/export_results.py derive-conf asv.conf.json "$conf" "$results"
          asv run "$head^!" --interleave-rounds --record-samples \
            --bench ray_ --machine {params.machine} \
            --config "$conf" --show-stderr
          python benchflow/export_results.py memory-save "$i" \
            --results "$results" --commit "$head" --machine {params.machine}
        done
        python benchflow/export_results.py memory-pair
        """


rule pr_compare:
    input:
        "benchflow/out/thresholds.json",
        "benchflow/out/machine.done",
    output:
        report="benchflow/out/pr_compare.txt",
        status="benchflow/out/pr_compare.status",
    params:
        base=config["base"],
        head=config["head"],
        rounds=config["rounds"],
        machine=config["machine"],
    shell:
        """
        set -euo pipefail
        . benchflow/env.sh
        base=$(git rev-parse {params.base})
        head=$(git rev-parse {params.head})
        log=benchflow/out/pr_asv.txt
        if [ "$base" = "$head" ]; then
          printf '%s\n' "PR gate requires distinct base and head revisions" > "$log"
          python benchflow/export_results.py gate \
            --base "$base" --head "$head" --machine {params.machine} \
            --asv-log "$log" --asv-status 1 \
            --report {output.report} --status {output.status}
        else
          conf=".asv-run-pr.conf.json"
          results="benchflow/out/results/pr"
          rm -rf "$results"
          python benchflow/export_results.py derive-conf asv.conf.json "$conf" "$results"
          set +e
          asv continuous {params.base} {params.head} --interleave-rounds \
            --record-samples -a rounds={params.rounds} --bench time_ \
            --machine {params.machine} --config "$conf" --show-stderr \
            > "$log" 2>&1
          asv_status=$?
          set -e
          python benchflow/export_results.py gate \
            --base "$base" --head "$head" --machine {params.machine} \
            --results "$results" \
            --asv-log "$log" --asv-status "$asv_status" \
            --thresholds benchflow/out/thresholds.json \
            --report {output.report} --status {output.status}
        fi
        """


rule pr_gate:
    input:
        "benchflow/out/pr_compare.txt",
        "benchflow/out/pr_compare.status",
    output:
        touch("benchflow/out/pr_gate.done"),
    shell:
        """
        set -euo pipefail
        status=$(tr -d '[:space:]' < benchflow/out/pr_compare.status)
        exit "$status"
        """
