import pickle
import json
import os

# Congif
MODEL_PATH = "price_model.pkl"
ENCODER_DIR = "encoders"

# Load model and encoders once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

encoders = {}
for col in ["make", "model", "year_range", "part_name", "part_condition"]:
    with open(os.path.join(ENCODER_DIR, f"{col}.json"), "r") as f:
        encoders[col] = json.load(f) # { "ford": 0, "volkswagen": 1, ... }

def encode(col, value):
    mapping = encoders[col]
    if value not in mapping:
        raise ValueError(f"Unknown value '{value}' for '{col}'. Valid: {list(mapping.keys())}")
    return mapping[value]

def predict_single(make, model_name, year_range, part_name, condition):
    """
    Predict price for one part + one condition.
    Returns a single float price.
    """
    features = [[
        encode("make", make),
        encode("model", model_name),
        encode("year_range", year_range),
        encode("part_name", part_name),
        encode("part_condition", condition),
    ]]
    return round(float(model.predict(features)[0]), 2)

def predict_part(make, model_name, year_range, part_name):
    """
    Predict all 3 price scenarios for one damaged part.
    Returns a dict with original_new, original_used, aftermarket prices.
    """
    return {
        "part_name": part_name,
        "original_new": predict_single(make, model_name, year_range, part_name, "original_new"),
        "original_used": predict_single(make, model_name, year_range, part_name, "original_used"),
        "aftermarket": predict_single(make, model_name, year_range, part_name, "aftermarket"),
    }

def predict_all_parts(make, model_name, year_range, damaged_parts):
    """
    Main inference function.
 
    Input:
        make          : str  — e.g. "ford"
        model_name    : str  — e.g. "fusion"
        year_range    : str  — e.g. "2013_2016"
        damaged_parts : list — e.g. ["front-bumper-dent", "Headlight-Damage", "bonnet-dent"]
 
    Output:
        list of dicts, one per part:
        [
            {
                "part_name":     "front-bumper-dent",
                "original_new":  250.0,
                "original_used": 70.0,
                "aftermarket":   45.0
            },
            ...
        ]
    """
    results = []
    for part in damaged_parts:
        result = predict_part(make, model_name, year_range, part)
        results.append(result)
    return results

# Example Usage
if __name__ == "__main__":

    # Simulate from input with 3 damaged parts
    make = "ford"
    model_name = "fusion"
    year_range = "2013_2016"
    damaged_parts = ["front-bumper-dent", "Headlight-Damage", "bonnet-dent"]

    results = predict_all_parts(make, model_name, year_range, damaged_parts)

    # Print results table
    print(f"\nVehicle: {make.title()} {model_name.title()} ({year_range})")
    print(f"{'Part':<30} {'Original New':>14} {'Original Used':>14} {'Aftermarket':>12}")
    print("-" * 74)
    for r in results:
        print(f"{r['part_name']:<30} ${r['original_new']:>12.2f}  ${r['original_used']:>12.2f}  ${r['aftermarket']:>10.2f}")
    
    total_new = sum(r['original_new'] for r in results)
    total_used = sum(r['original_used'] for r in results)
    total_aftermarket = sum(r['aftermarket'] for r in results)

    print("-" * 74)