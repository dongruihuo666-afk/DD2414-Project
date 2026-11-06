"""Check DDP loss and gradients against the unmodified single-GPU closure."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import depth_pretraining_objective as objective


class TinyBackbone(nn.Module):
    def __init__(self, depth_mode):
        super().__init__()
        self.depth_mode = depth_mode
        self.feature = nn.Parameter(torch.randn(2, 3))
        self.depth = nn.Parameter(torch.randn(1, 1, 2, 3))
        self.image_head, self.bev_head, self.motion_head = [nn.Linear(3, 3) for _ in range(3)]

    def features(self, inputs):
        return self.feature, self.feature * 2, self.depth * 1 if self.depth_mode != 'none' else None


class DepthObjectiveTests(unittest.TestCase):
    def test_all_three_depth_objectives_and_student_gradients_match_legacy(self):
        source = Path(__file__).with_name('train_multisweep_teacher.py').read_text()
        function = next(node for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef) and node.name == 'pretrain_loss')
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                       '<actual single-GPU pretrain_loss>', 'exec')
        mse = lambda a, b, *rest: (a - b).square().mean()
        inputs = (None, (torch.tensor([7]),))
        target = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
        dependencies = dict(
            recent_radar_target_batch=lambda batch: batch,
            extract_teacher_features_batched=lambda *a, **k: target,
            build_targets_from_features_batched=lambda *a: (target, torch.ones(2, 3)),
            prepare_bevcar_inputs=lambda *a: inputs,
            apply_radar_velocity_mode=lambda value, mode: value,
            image_cosine_loss=mse, semantic_loss_batched=mse,
            build_motion_targets_batched=lambda *a: (target, torch.ones(2, 3), None),
            motion_loss_batched=lambda a, b, c: (mse(a, b), 6),
            projected_radar_depth=lambda *a: target,
            sparse_depth_loss=lambda a, b: (mse(a, b), dict(depth_anchors=6)),
            relative_depth_target=lambda *a: target * 2,
            relative_depth_loss=mse)
        batch = [torch.zeros(1, 1, 2, 3)] + [None] * 15 + [torch.ones(1, 1, 18, 2)]
        args = SimpleNamespace(teacher_resolution=[448, 798], depth_teacher_resolution=[448, 798],
            velocity_mode='full', image_weight=.2, bev_weight=.8, motion_weight=.5,
            depth_weight=.2, relative_depth_weight=.05)
        for mode in ['none', 'radar', 'radar-dav2']:
            with self.subTest(depth_mode=mode), patch.multiple(objective, **dependencies):
                torch.manual_seed(12)
                old = TinyBackbone(mode)
                new = copy.deepcopy(old)
                teacher = object()
                depth_teacher = object() if mode == 'radar-dav2' else None
                namespace = dict(dependencies, torch=torch, backbone=old, teacher=teacher,
                    depth_teacher=depth_teacher, args=args, device=torch.device('cpu'),
                    model_inputs=lambda *a: inputs)
                exec(code, namespace)
                previous = namespace['pretrain_loss'](batch)
                current = objective.pretraining_loss(batch, objective.PretrainingForward(new),
                    teacher, depth_teacher, args, torch.device('cpu'))
                self.assertEqual(previous[2], current[2])
                self.assertEqual(previous[1].keys(), current[1].keys())
                torch.testing.assert_close(previous[0], current[0], rtol=0, atol=0)
                previous[0].backward()
                current[0].backward()
                for (name, first), (_, second) in zip(old.named_parameters(), new.named_parameters()):
                    if first.grad is None:
                        self.assertIsNone(second.grad, name)
                    else:
                        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0, msg=name)


if __name__ == '__main__':
    unittest.main()
