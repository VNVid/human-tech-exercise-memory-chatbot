import statistics

def parse_results(path):
    stats = {}          # { 'D1': [(x1,y1), (x2,y2), (x3,y3)], ... }
    with open(path, 'r') as f:
        current = None
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('D'):
                current = line
                stats[current] = []
            else:
                x, y = map(int, line.split())
                stats[current].append((x, y))
    return stats

def compute_and_report(stats):
    overall_x = overall_y = 0
    perfect_dialogues = missed = 0

    # for the new stats:
    total_runs = perfect_runs = 0
    worst_run = (1.0, None, None, None, None)
    # tuple: (ratio, dialogue, run_index, x, y)
    mean_rates = {}  # dialogue → mean rate

    print("Per-dialogue statistics:")
    for d, runs in stats.items():
        # compute per-run ratios, track perfect runs & worst run
        ratios = []
        for i, (x, y) in enumerate(runs, start=1):
            r = x / y if y else 0
            ratios.append(r)
            total_runs += 1
            if x == y:
                perfect_runs += 1
            if r < worst_run[0]:
                worst_run = (r, d, i, x, y)

        # per-dialogue aggregates
        mean_r = statistics.mean(ratios)
        std_r  = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        mean_rates[d] = mean_r
        tot_x = sum(x for x, _ in runs)
        tot_y = sum(y for _, y in runs)
        overall_x += tot_x
        overall_y += tot_y

        if all(x == y for x, y in runs):
            perfect_dialogues += 1
        if all(x == 0 for x, _ in runs):
            missed += 1

        print(f"{d}:")
        print(f"    Runs: {[f'{x}/{y} ({x/y:.2f})' for x,y in runs]}")
        print(f"    Mean rate = {mean_r:.2f},  Stddev = {std_r:.2f}")

    # pick the worst-performing dialogue by mean rate
    worst_dialogue = min(mean_rates, key=mean_rates.get)
    worst_dialogue_rate = mean_rates[worst_dialogue]

    overall_rate = overall_x / overall_y if overall_y else 0
    perfect_dialogue_rate = perfect_dialogues / len(stats) if stats else 0
    perfect_run_rate = perfect_runs / total_runs if total_runs else 0

    print("\nOverall performance:")
    print(f"Global detection rate = {overall_x}/{overall_y} ({overall_rate:.2f})")
    print(f"Perfect dialogues = {perfect_dialogues}/{len(stats)} ({perfect_dialogue_rate:.2%})")
    print(f"Perfect runs = {perfect_runs}/{total_runs} ({perfect_run_rate:.2%})")

    r, d, idx, x, y = worst_run
    print(f"The worst run = {d}, run #{idx}: {x}/{y} ({r:.2f})")

    print(f"Worst dialogue = {worst_dialogue} (mean rate = {worst_dialogue_rate:.2f})")

if __name__ == '__main__':
    stats = parse_results('results.txt')
    compute_and_report(stats)