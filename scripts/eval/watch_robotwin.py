#!/usr/bin/env python3
"""Show live success rate and recent episodes without producing files."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


def render(value):
    count=value["completed_episodes"]
    rate=value["success_rate"]
    rate_text=f"{rate:.1%}" if rate is not None else "pending"
    lines=[f'{value["task"]} | {value["status"]} | {count}/{value["planned_episodes"]} completed',
           f'Success {value["successes"]}  Fail {value["failures"]}  SR {rate_text} (completed only)',
           f'Updated {value["updated_at"]}', ""]
    for gpu,worker in value["workers"].items():
        lines.append(f'GPU{gpu}: {worker["completed"]}/{worker["planned"]}  '
                     f'success={worker["successes"]}  {worker["status"]}  seed={worker.get("current_seed")}')
    lines += ["", "Recent episodes: seed / GPU / result"]
    recent=sorted(value["results"],key=lambda r:r.get("completed_at",""))[-12:]
    lines += [f'{r["seed"]} / {r["gpu"]} / {"SUCCESS" if r["success"] else "FAIL"}' for r in recent]
    if value.get("error"):
        lines += ["",value["error"]]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("result",type=Path)
    p.add_argument("--once",action="store_true")
    p.add_argument("--interval",type=float,default=1)
    args=p.parse_args()
    if args.interval<=0:p.error("--interval must be positive")
    while True:
        try:
            value=json.loads(args.result.read_text())
        except FileNotFoundError:
            value=None
        if sys.stdout.isatty() and not args.once:
            print("\033[2J\033[H",end="")
        print(render(value) if value else f"Waiting for {args.result}",flush=True)
        if args.once or (value and value["status"]!="running"):
            break
        time.sleep(args.interval)


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
