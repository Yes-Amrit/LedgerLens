import argparse
import json
from agent.graph import run_graph

def main():
    parser = argparse.ArgumentParser(description="Run LedgerLens Agent")
    parser.add_argument("--query", type=str, required=True, help="Natural language query for the agent")
    args = parser.parse_args()
    
    print(f"Running agent with query: {args.query}\n")
    
    result = run_graph(args.query)
    
    summary = result.get("execution_summary", {})
    print("Execution Summary:")
    print(json.dumps(summary, indent=2))
    
    print("\nFull Result State Keys:")
    print(list(result.keys()))
    
if __name__ == "__main__":
    main()
