"""Checks for the new distribution target, independent of pretrained weights."""
import unittest

import torch
from transformers import AutoTokenizer
from pathlib import Path

from submodules.recap_model import tokenize_task_prompts, two_hot_loss
from submodules.recap_workflow import optimizer_groups, run_recap_epoch

ROOT = Path(__file__).resolve().parents[1]


class RecapTargetTests(unittest.TestCase):
    def test_task_tokens_right_padding_and_validity_mask(self):
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / 'models/gemma-3-270m', local_files_only=True)
        prompts = ['Pick_up the red cup. ', 'Pick up the red cup and hang it on the cup holder.']
        ids, mask = tokenize_task_prompts(tokenizer, prompts, 50, 'cpu')
        self.assertEqual(tuple(ids.shape), (2, 50))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(ids[0, :mask[0].sum()].tolist(),
                         tokenizer.encode('Task: pick up the red cup.', add_special_tokens=True))
        self.assertTrue(mask[0, :mask[0].sum()].all())
        self.assertFalse(mask[0, mask[0].sum():].any())
        self.assertTrue(torch.eq(ids[0, ~mask[0]], tokenizer.pad_token_id).all())
        self.assertGreater(int(mask[1].sum()), int(mask[0].sum()))

    def test_adjacent_atom_projection_and_gradients(self):
        atoms = torch.tensor([-1.0, -0.5, 0.0])
        logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
        target = torch.tensor([-0.25])
        loss = two_hot_loss(logits, target, atoms)
        log_probs = torch.log_softmax(logits, dim=-1)
        expected = -(log_probs[0, 1] + log_probs[0, 2]) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_support_endpoints_and_clamping(self):
        atoms = torch.linspace(-1, 0, 201)
        logits = torch.randn(3, 201)
        targets = torch.tensor([-3.0, 0.0, 2.0])
        loss = two_hot_loss(logits, targets, atoms)
        logs = torch.log_softmax(logits, dim=-1)
        expected = -(logs[0, 0] + logs[1, -1] + logs[2, -1]) / 3
        torch.testing.assert_close(loss, expected)


class RecapTrainingTests(unittest.TestCase):
    def test_independent_optimizer_groups_respect_frozen_parameters(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = torch.nn.Linear(1, 1)
                self.gemma3 = torch.nn.Linear(1, 1)
                self.expert = torch.nn.Linear(1, 1)
                self.cls_embedding = torch.nn.Parameter(torch.ones(1))

        model = TinyModel()
        config = dict(vision_lr=1e-5, gemma_lr=2e-5, expert_lr=3e-5)
        groups = optimizer_groups(model, config)
        self.assertEqual([(group['name'], group['lr']) for group in groups],
                         [('vision', 1e-5), ('gemma3', 2e-5), ('expert', 3e-5)])
        model.vision_tower.requires_grad_(False)
        groups = optimizer_groups(model, config)
        self.assertEqual([group['name'] for group in groups], ['gemma3', 'expert'])
        self.assertEqual(sum(len(group['params']) for group in groups),
                         sum(parameter.requires_grad for parameter in model.parameters()))

    def test_gradient_accumulation_matches_one_full_batch(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.2))
                self.register_buffer('atoms', torch.linspace(-1, 0, 3))

            def forward(self, images, prompts, return_distribution=False, image_masks=None):
                x = images['observation.images.cam2'].reshape(-1)
                logits = torch.stack((x * self.weight, -x * self.weight,
                                      x * self.weight * 2), dim=-1)
                values = (logits.softmax(-1) * self.atoms).sum(-1, keepdim=True)
                return values, logits

        def batch(inputs, targets):
            return dict(images={'observation.images.cam2': torch.tensor(inputs)},
                        target_values=torch.tensor(targets), prompt=['task'] * len(inputs),
                        image_masks={'observation.images.cam2': torch.ones(len(inputs), dtype=torch.bool)})

        micro = [batch([1.0], [-0.8]), batch([2.0, 3.0], [-0.4, -0.1])]
        full = [batch([1.0, 2.0, 3.0], [-0.8, -0.4, -0.1])]
        results = []
        for loader, accumulation in ((micro, 2), (full, 1)):
            model = TinyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            metrics = run_recap_epoch(model, loader,
                                      dict(precision='fp32', gradient_accumulation_steps=accumulation,
                                           clip_grad_norm=100.0), torch.device('cpu'),
                                      optimizer, scheduler)
            self.assertEqual(metrics['steps'], 1)
            self.assertEqual(metrics['samples'], 3)
            results.append((model.weight.detach(), metrics['ce']))
        torch.testing.assert_close(results[0][0], results[1][0])
        self.assertAlmostEqual(results[0][1], results[1][1], places=6)


if __name__ == '__main__':
    unittest.main()
