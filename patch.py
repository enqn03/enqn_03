import torch
import torch.nn.functional as F
from train_a_only_baseline import raw_coordinate_from_model_index
from typing import Any

def make_fusion_candidate_evaluator(model: torch.nn.Module, dataset: Any, device: torch.device, evaluation: dict[str, Any], max_samples: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    candidates: list[dict[str, Any]] = []
    endpoint_statuses: list[dict[str, Any]] = []
    
    top_k = int(evaluation["top_k_candidates_per_endpoint"])
    kernel_size = int(evaluation["local_maximum_kernel_size"])
    plateau_range_atol = float(evaluation["spatial_plateau_range_atol"])
    top_score_tie_atol = float(evaluation["top_score_tie_atol"])
    top_score_tie_fraction_max = float(evaluation["top_score_tie_fraction_max"])

    with torch.no_grad():
        sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for index in range(sample_count):
            sample = dataset[index]
            history_a = sample["model_input_history_a"].unsqueeze(0).to(device=device, dtype=torch.float32)
            history_b = sample["model_input_history_b"].unsqueeze(0).to(device=device, dtype=torch.float32)
            prediction = model(history_a, history_b).cpu()
            
            # Simple local_maximum_candidates logic inline
            candidate_map = prediction[0, 0]
            prediction_min, prediction_max = float(candidate_map.min().item()), float(candidate_map.max().item())
            spatial_range = prediction_max - prediction_min
            
            z = int(sample["endpoint_layer_z"])
            
            top_score_tie_pixel_count = int(torch.isclose(candidate_map, candidate_map.max(), rtol=0.0, atol=top_score_tie_atol).sum().item())
            top_score_tie_fraction = top_score_tie_pixel_count / int(candidate_map.numel())
            
            if spatial_range <= plateau_range_atol:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_spatial_plateau"})
                continue
            if top_score_tie_fraction > top_score_tie_fraction_max:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_top_score_plateau"})
                continue
                
            pooled = F.max_pool2d(candidate_map[None, None], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[0, 0]
            maxima = candidate_map == pooled
            finite_maxima = maxima & torch.isfinite(candidate_map)
            
            scores = candidate_map.masked_fill(~finite_maxima, -torch.inf).flatten()
            count = min(top_k, int(torch.isfinite(scores).sum().item()))
            
            if count == 0:
                endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "withheld_no_local_maximum"})
                continue
                
            values, flat_indices = torch.topk(scores, k=count)
            width = candidate_map.shape[1]
            metadata = sample["metadata"]
            
            emitted = 0
            for rank, (score, flat_index) in enumerate(zip(values.tolist(), flat_indices.tolist()), start=1):
                y_model, x_model = divmod(int(flat_index), int(width))
                x_raw, y_raw = raw_coordinate_from_model_index(x_model, y_model, metadata)
                candidates.append({
                    "x_pixel": x_raw, "y_pixel": y_raw, "layer_z": z,
                    "score": float(score), "status": "candidate", "stage": "fusion"
                })
                emitted += 1
                
            endpoint_statuses.append({"endpoint_layer_z": z, "candidate_status": "emitted", "count": emitted})

    return candidates, endpoint_statuses
