import sys

with open("src/train_a_b_fusion_temporal_difference_v1.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("def make_fusion_candidate_evaluator"):
        start_idx = i
    elif start_idx != -1 and line.startswith("def main()"):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    with open("patch.py", "r") as f:
        patch_code = f.read()
    
    # We don't want the patch's imports to be repeated, so we'll just insert the function
    # But wait, raw_coordinate_from_model_index needs to be imported.
    # Let's just insert the patch_code before make_fusion_candidate_evaluator.
    # Actually, let's just replace the function body.
    
    new_content = "".join(lines[:start_idx]) + patch_code + "\n\n" + "".join(lines[end_idx:])
    
    # Add the import at the top if not exists
    if "raw_coordinate_from_model_index" not in new_content:
        new_content = new_content.replace("from train_a_only_baseline import (", "from train_a_only_baseline import (\n    raw_coordinate_from_model_index,")
        
    with open("src/train_a_b_fusion_temporal_difference_v1.py", "w") as f:
        f.write(new_content)
        
    print("Patch applied successfully.")
else:
    print("Could not find bounds.")
