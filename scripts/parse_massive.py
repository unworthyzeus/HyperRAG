import re
import json

def parse_results(filename):
    results = []
    current_model = None
    baseline_mrr = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Detect Model
        if "# MODEL:" in line:
            parts = line.split("MODEL:")[1].strip().split("(")
            current_model = parts[0].strip()
            baseline_mrr = None # Reset for new model
            
        # Detect Engine run
        if "[1/5] ST Baseline" in line:
            current_engine = "ST Baseline"
        elif "[2/5] CrossPolytope" in line:
            current_engine = "CrossPolytope L1"
        elif "Hybrid Radial" in line and "[3/5]" in line:
            current_engine = "Hybrid Radial (a=0.1)"
        elif "Hybrid Radial" in line and "[4/5]" in line:
            current_engine = "Hybrid Radial (a=0.2)"
        elif "Hybrid Radial" in line and "[5/5]" in line:
            current_engine = "Hybrid Radial (a=0.3)"
            
        # Detect MRR
        if "MRR:" in line and current_model:
            # Extract MRR, H@1, H@5
            # Format: MRR: 0.7421, H@1: 61.9%, H@5: 90.4%
            try:
                mrr_match = re.search(r"MRR: ([\d\.]+)", line)
                if mrr_match:
                    mrr = float(mrr_match.group(1))
                    
                    if "ST Baseline" in current_engine:
                        baseline_mrr = mrr
                    
                    results.append({
                        "model": current_model,
                        "engine": current_engine,
                        "mrr": mrr,
                        "baseline_diff": ((mrr - baseline_mrr)/baseline_mrr * 100) if baseline_mrr and "ST Baseline" not in current_engine else 0.0
                    })
            except:
                pass

    return results

data = parse_results("massive_output.txt")
print(json.dumps(data, indent=2))
with open("results/massive_summary.json", "w") as f:
    json.dump(data, f, indent=2)
