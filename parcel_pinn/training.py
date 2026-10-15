import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def seed_all(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit(model, loss_function, epochs, lr, out, *, decay=.25, fail_at=None,
        polish_steps=0, polish_function=None):
    """Retain best finite CHECKED checkpoint; always return on numerical breakdown.

    Programming and I/O errors are recorded by pipeline, not called nonconvergence.
    The objective uses a fixed audit batch, so checkpoint comparisons are meaningful.
    Training uses fresh collocation points. No reference values enter the objective.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay**(1/max(epochs, 1)))
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_step = float('inf'), 0
    rows, reason = [], 'budget_exhausted'
    started = time.monotonic()
    try:
        initial, _ = loss_function(audit=True)
        best_loss = float(initial.detach())
        if not np.isfinite(best_loss):
            raise FloatingPointError('initial residual is nonfinite')
        rows.append(dict(step=0, loss=best_loss, audit_loss=best_loss, lr=lr))
        for step in range(1, epochs+1):
            opt.zero_grad(set_to_none=True)
            loss, components = loss_function(audit=False)
            if fail_at == step:
                loss = loss*float('nan')
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'nonfinite gradient at step {step}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step(); scheduler.step()
            if step == 1 or step == epochs or step % max(1, epochs//50) == 0:
                audit, _ = loss_function(audit=True)
                value = float(audit.detach())
                if not np.isfinite(value):
                    raise FloatingPointError(f'nonfinite audit at step {step}')
                if value < best_loss:
                    best_loss, best_step = value, step
                    best_state = copy.deepcopy(model.state_dict())
                rows.append(dict(step=step, loss=float(loss.detach()), audit_loss=value,
                                 lr=opt.param_groups[0]['lr'],
                                 **{k: float(v.detach()) for k, v in components.items()}))
        if epochs == 0 and polish_steps == 0:
            reason = 'no_training_requested'
        if polish_steps and polish_function is not None:
            model.load_state_dict(best_state)
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=1,
                                      history_size=30, line_search_fn='strong_wolfe')
            for k in range(1,polish_steps+1):
                def closure():
                    lbfgs.zero_grad(set_to_none=True)
                    loss,_=polish_function()
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite L-BFGS objective')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite L-BFGS gradient')
                    return loss
                train_value=lbfgs.step(closure)
                audit,_=loss_function(audit=True)
                value=float(audit.detach())
                if not np.isfinite(value):raise FloatingPointError('nonfinite L-BFGS audit')
                if value<best_loss:
                    best_loss,best_step=value,epochs+k
                    best_state=copy.deepcopy(model.state_dict())
                rows.append(dict(step=epochs+k,loss=float(train_value.detach()),audit_loss=value,lr=.8,optimizer='L-BFGS'))
    except (FloatingPointError, RuntimeError) as exc:
        reason = f'training_error: {type(exc).__name__}: {exc}'
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history.to_csv(out/'training.csv', index=False)
    state = dict(termination=reason, best_step=best_step, best_audit_loss=best_loss,
                 completed_steps=int(rows[-1]['step']) if rows else 0,
                 checkpoint_kind='trained' if best_step else 'initialized',
                 seconds=time.monotonic()-started)
    torch.save({'state_dict': model.state_dict(), 'training': state}, out/'checkpoint.pt')
    (out/'training_status.json').write_text(json.dumps(state, indent=2))
    return state, history


def train_channel(model, out, fail_at=None):
    c = model.cfg
    generator = torch.Generator().manual_seed(c.seed+101)
    def batch(gen,nt=None,ns=None):
        nt,ns=nt or c.time_batch,ns or c.space_batch
        t = torch.rand(nt, 1, generator=gen, dtype=torch.float64).square()*c.final_time
        xyz = torch.rand(nt, ns, c.dimension, generator=gen, dtype=torch.float64)
        return t, xyz*xyz.new_tensor(c.lengths)
    audit = batch(torch.Generator().manual_seed(19481),max(c.time_batch,7),max(c.space_batch,96))
    audit[0][0]=0.;audit[0][-1]=c.final_time
    polish = batch(torch.Generator().manual_seed(8516),max(c.time_batch,6),max(c.space_batch,96))
    polish[0][0]=0.;polish[0][-1]=c.final_time
    def objective(audit=False):
        return model.loss(*(audit_batch if audit else batch(generator)))
    audit_batch = audit
    return fit(model, objective, c.epochs, c.lr, out, decay=c.learning_rate_decay, fail_at=fail_at,
               polish_steps=c.lbfgs_steps,polish_function=lambda:model.loss(*polish))
