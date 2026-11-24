import torch
from transformers import AutoTokenizer
from model import PhysicalBudgetLogitsProcessor

tok = AutoTokenizer.from_pretrained('t5-small')
budgets = torch.tensor([5.0])
proc = PhysicalBudgetLogitsProcessor(tokenizer=tok, budgets=budgets, eos_token_id=1, penalty_weight=10.0, margin_buffer=0.0, hard_cutoff_excess=2.0)

# Simulate generation steps
tok_ids = tok.encode('testing words now', add_special_tokens=False)
print('Tokens for input:', tok_ids)

seq = [0]
for t in tok_ids:
    seq.append(t)
    curr_input = torch.tensor([seq])
    sc = torch.zeros(1, 32128)
    mod_sc = proc(curr_input, sc)
    dec = tok.decode(curr_input[0], skip_special_tokens=True)
    print(f'Seq: {seq}, Decoded: \"{dec}\" (chars: {len(dec)}), Budget: 5.0')
    print(f'  EOS score: {mod_sc[0, 1].item():.1f}, Token 0 score: {mod_sc[0, 0].item():.1f}')

