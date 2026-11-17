"""
renloc_qe/worker.py
Standard I/O JSON-RPC 2.0 loop for isolated Quality Estimation.
"""
import sys
import json
import traceback
from .scorer import QualityScorer

def main():
    scorer = QualityScorer()
    # Output ready signal to stderr
    print("[*] renloc_qe worker initialized and listening on stdin.", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": "pong"}
            elif method == "score_batch":
                items = params.get("items", [])
                scores = scorer.score_batch(items)
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"scores": scores}}
            elif method == "shutdown":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": "bye"}
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                break
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method {method} not found"}
                }
        except Exception as e:
            tb = traceback.format_exc()
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e), "data": tb}
            }
        
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
