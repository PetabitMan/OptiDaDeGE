import json
from tqdm import tqdm

# Pfade anpassen falls nötig
ANSWERED_LAWS_FILE   = "answered_laws.json"
ENRICHED_CASES_FILE  = "enriched_cases.json"
OUTPUT_METRICS_FILE  = "evaluation_results_for_answers.json"

def load_data():
    with open(ANSWERED_LAWS_FILE,  encoding="utf-8") as f:
        answered = json.load(f)
    with open(ENRICHED_CASES_FILE, encoding="utf-8") as f:
        enriched = json.load(f)
    return answered, enriched

def build_gt_map(enriched):
    # case id -> ground-truth Referenzen
    return {
        case["id"]: set(case.get("simple_refs", []))
        for case in enriched
        if "id" in case and case.get("simple_refs")
    }

def compute_f1_for_threshold(answered, gt_map, abs_th=None, pct_th=None):
    tp = fp = fn = 0
    for case in answered:
        cid = case.get("case_id")
        gt  = gt_map.get(cid, set())
        if not gt:
            continue

        sys_set = set()
        for law in case.get("laws", []):
            ref     = law["law_reference"]
            answers = [tbm.get("answer") for tbm in law.get("tatbestandsmerkmale", [])]
            yes     = sum(1 for a in answers if a == "")
            total   = len(answers)

            meets = False
            if abs_th is not None:
                meets = yes >= abs_th
            elif pct_th is not None and total > 0:
                meets = (yes / total) >= pct_th

            if meets:
                sys_set.add(ref)

        tp += len(sys_set & gt)
        fp += len(sys_set - gt)
        fn += len(gt - sys_set)

    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
    return precision, recall, f1

def evaluate_thresholds(answered, gt_map):
    # max TBMs ermitteln
    max_tbms = 0
    for case in answered:
        for law in case.get("laws", []):
            n = len(law.get("tatbestandsmerkmale", []))
            if n > max_tbms:
                max_tbms = n

    abs_thresholds = list(range(0, max_tbms+1))
    pct_thresholds = [i/10 for i in range(0,11)]  # 0.1 … 0.9

    best_abs = {"threshold": None, "precision":0, "recall":0, "f1":0}
    best_pct = {"threshold": None, "precision":0, "recall":0, "f1":0}
    details  = {"absolute": [], "percent": []}

    # absolute Schwellen
    for th in abs_thresholds:
        p,r,f = compute_f1_for_threshold(answered, gt_map, abs_th=th)
        details["absolute"].append({"threshold":th, "precision":p, "recall":r, "f1":f})
        if f > best_abs["f1"]:
            best_abs = {"threshold":th, "precision":p, "recall":r, "f1":f}

    # prozentuale Schwellen
    for th in pct_thresholds:
        p,r,f = compute_f1_for_threshold(answered, gt_map, pct_th=th)
        details["percent"].append({"threshold":th, "precision":p, "recall":r, "f1":f})
        if f > best_pct["f1"]:
            best_pct = {"threshold":th, "precision":p, "recall":r, "f1":f}

    return best_abs, best_pct, details

def main():
    answered, enriched = load_data()
    gt_map = build_gt_map(enriched)

    best_abs, best_pct, details = evaluate_thresholds(answered, gt_map)

    result = {
        "best_absolute": best_abs,
        "best_percent":  best_pct,
        "all_details":  details
    }

    with open(OUTPUT_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("✅ Evaluation complete. Results saved to", OUTPUT_METRICS_FILE)
    print(f"Best absolute threshold: {best_abs['threshold']}   F1={best_abs['f1']:.2f}")
    print(f"Best percent  threshold: {best_pct['threshold']*100:.0f}%  F1={best_pct['f1']:.2f}")

if __name__=="__main__":
    main()
