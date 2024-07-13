from typing import Dict
import torch
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # aux_loss = outputs.aux_loss
        polarization_loss = outputs.polarization_loss
        ld_loss = outputs.ld_loss

        router_ce_loss = outputs.router_ce_loss

        if router_ce_loss != None:
            prefix = "router_ce"
            lang_mask = inputs['langs']
            with torch.no_grad():
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # 24 x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                mask = lang_mask.reshape(-1).bool().expand(probs.size()[:2])
                probs = probs[mask].to(torch.float).reshape(mask.size(0), -1, probs.size(-1)) # 24 x n x e
                probs = probs[:, :, 0] # 24 x n
                score_expert0 = torch.mean(probs, dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = router_ce_loss.item()
                logs["old_lang_expert0_score"] = " ".join([str(round(k, 2)) for k in score_expert0.tolist()])
                self.log(logs)

        if polarization_loss != None:
            prefix = "polar_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                mask = mask.unsqueeze(0).expand(24, mask.size(0)) # 24 x n
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # 24 x n x 2
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                max_probs = torch.max(probs, dim=-1).values # 24 x n
                max_probs = max_probs[mask.bool()].reshape(24, -1)
                tokens_over_bound = max_probs >= 0.9
                scores = torch.mean(tokens_over_bound.float(), dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = polarization_loss.item()
                logs["experts_info_per_layer"] = " ".join([str(round(k, 2)) for k in scores.tolist()])
                self.log(logs)

        if ld_loss != None:
            prefix = "ld_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                mask = mask.unsqueeze(0).expand(24, mask.size(0)) # 24 x n
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # 24 x n x 2
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                expert_index = torch.argmax(probs, dim=-1) # 24 x n
                expert_index = expert_index[mask.bool()].reshape(24, -1)
                scores = torch.mean(expert_index.float(), dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = ld_loss.item()
                logs["experts_info_per_layer"] = " ".join([str(round(k, 2)) for k in scores.tolist()])
                self.log(logs)

        if self.main_loss_logged: self.main_loss_logged = False
        return loss