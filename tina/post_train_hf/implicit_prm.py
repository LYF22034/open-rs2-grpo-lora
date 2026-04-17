"""
Implicit Process Reward Model (PRM) for PRIME.
Models live on CPU, loaded to GPU only when needed (time-sharing).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from trl.trainer.utils import selective_log_softmax


class ImplicitPRM:

    def __init__(self, model_name_or_path, beta=0.05, lr=1e-6, device="cuda:0", torch_dtype=torch.bfloat16):
        self.beta = beta
        self.device = device
        self.dtype = torch_dtype

        # Load PRM on CPU (save GPU memory)
        print(f"[ImplicitPRM] Loading PRM from {model_name_or_path} (on CPU)...")
        self.prm_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2", use_cache=False,
        )
        self.prm_model.train()
        # Enable gradient checkpointing to reduce activation memory
        self.prm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

        # Load Reference on CPU
        print(f"[ImplicitPRM] Loading Reference from {model_name_or_path} (on CPU)...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2", use_cache=False,
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer (works on CPU params, steps happen after .to(device))
        self.optimizer = torch.optim.AdamW(self.prm_model.parameters(), lr=lr, weight_decay=0.0)

        self.total_updates = 0
        prm_params = sum(p.numel() for p in self.prm_model.parameters())
        print(f"[ImplicitPRM] Ready. beta={beta}, lr={lr}, params={prm_params/1e6:.1f}M")

    def _to_gpu(self):
        self.prm_model.to(self.device)
        self.ref_model.to(self.device)

    def _to_cpu(self):
        self.prm_model.to("cpu")
        self.ref_model.to("cpu")
        torch.cuda.empty_cache()

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits
        logits = logits[:, :-1, :]
        ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, ids)

    @torch.no_grad()
    def compute_token_rewards(self, input_ids, attention_mask, logits_to_keep):
        """Compute token-level implicit rewards. Loads to GPU, computes, unloads."""
        self._to_gpu()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Process in chunks to save memory
        chunk_size = 4
        B = input_ids.size(0)
        all_rewards = []

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]

            prm_logps = self._get_per_token_logps(self.prm_model, chunk_ids, chunk_mask, logits_to_keep)
            ref_logps = self._get_per_token_logps(self.ref_model, chunk_ids, chunk_mask, logits_to_keep)
            rewards = self.beta * (prm_logps - ref_logps)
            all_rewards.append(rewards.cpu())
            del prm_logps, ref_logps, rewards
            torch.cuda.empty_cache()

        self._to_cpu()
        return torch.cat(all_rewards, dim=0)  # (B, C) on CPU

    def update(self, input_ids, attention_mask, completion_mask, accuracy, logits_to_keep, micro_batch_size=2):
        """Update PRM with CE loss. Loads to GPU, updates, unloads."""
        self._to_gpu()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        completion_mask = completion_mask.to(self.device).float()
        accuracy = accuracy.to(self.device).float()

        B = input_ids.size(0)
        n_micro = (B + micro_batch_size - 1) // micro_batch_size
        total_loss = 0.0

        self.optimizer.zero_grad()

        for start in range(0, B, micro_batch_size):
            end = min(start + micro_batch_size, B)
            mb_ids = input_ids[start:end]
            mb_mask = attention_mask[start:end]
            mb_cmask = completion_mask[start:end]
            mb_acc = accuracy[start:end]

            # PRM forward (with grad)
            prm_logps = self._get_per_token_logps(self.prm_model, mb_ids, mb_mask, logits_to_keep)

            # Ref forward (no grad)
            with torch.no_grad():
                ref_logps = self._get_per_token_logps(self.ref_model, mb_ids, mb_mask, logits_to_keep)

            token_rewards = self.beta * (prm_logps - ref_logps)
            response_reward = (token_rewards * mb_cmask).sum(dim=1)

            pred_prob = torch.sigmoid(response_reward.float())
            ce_loss = F.binary_cross_entropy(pred_prob, mb_acc)
            (ce_loss / n_micro).backward()

            total_loss += ce_loss.item()

            del prm_logps, ref_logps, token_rewards, response_reward, pred_prob, ce_loss
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(self.prm_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.total_updates += 1

        # Quick accuracy check
        with torch.no_grad():
            all_preds = []
            for start in range(0, B, micro_batch_size):
                end = min(start + micro_batch_size, B)
                p = self._get_per_token_logps(self.prm_model, input_ids[start:end], attention_mask[start:end], logits_to_keep)
                r = self._get_per_token_logps(self.ref_model, input_ids[start:end], attention_mask[start:end], logits_to_keep)
                rw = (self.beta * (p - r) * completion_mask[start:end]).sum(dim=1)
                all_preds.append((rw > 0).float().cpu())
                del p, r, rw
            preds = torch.cat(all_preds)
            prm_acc = (preds == accuracy.cpu()).float().mean().item()

        self._to_cpu()

        return {
            "prm/ce_loss": total_loss / n_micro,
            "prm/classification_acc": prm_acc,
            "prm/total_updates": self.total_updates,
        }

    def save_checkpoint(self, path):
        self.prm_model.save_pretrained(path)
        print(f"[ImplicitPRM] Saved to {path}")
