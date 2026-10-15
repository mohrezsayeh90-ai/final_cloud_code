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
        polish_steps=0, polish_function=None, max_seconds=None, audit_every=None,
        save_every_audit=False):
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
    def expired():
        return max_seconds is not None and time.monotonic()-started >= max_seconds

    def persist(termination):
        state = dict(termination=termination, best_step=best_step, best_audit_loss=best_loss,
                     completed_steps=int(rows[-1]['step']) if rows else 0,
                     checkpoint_kind='trained' if best_step else 'initialized',
                     seconds=time.monotonic()-started)
        # An interrupted worker leaves either the previous checkpoint or a complete new one.
        temporary = out/'checkpoint.tmp'
        torch.save({'state_dict': best_state, 'training': state}, temporary)
        temporary.replace(out/'checkpoint.pt')
        pd.DataFrame(rows).to_csv(out/'training.csv', index=False)
        (out/'training_status.json').write_text(json.dumps(state, indent=2))
        return state
    try:
        initial, _ = loss_function(audit=True)
        best_loss = float(initial.detach())
        if not np.isfinite(best_loss):
            raise FloatingPointError('initial residual is nonfinite')
        rows.append(dict(step=0, loss=best_loss, audit_loss=best_loss, lr=lr))
        if save_every_audit:
            persist('running')
        for step in range(1, epochs+1):
            if expired():
                reason = 'time_budget_exhausted'
                break
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
            opt.step()
            if max_seconds is None:
                scheduler.step()
            else:
                # The step count is unknown in a wall-clock run. Decay across
                # the allotted time, rather than across a million nominal steps.
                fraction = min(1., (time.monotonic()-started)/max(max_seconds, 1e-9))
                for group in opt.param_groups:
                    group['lr'] = lr*decay**fraction
            if step == 1 or step == epochs or step % (audit_every or max(1, epochs//50)) == 0 or expired():
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
                if save_every_audit:
                    persist('running')
                if max_seconds is not None:
                    print(f"step={step} audit={value:.5g} elapsed={time.monotonic()-started:.0f}s", flush=True)
        if epochs == 0 and polish_steps == 0:
            reason = 'no_training_requested'
        if expired():
            reason = 'time_budget_exhausted'
        if polish_steps and polish_function is not None and not expired():
            model.load_state_dict(best_state)
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=1,
                                      history_size=30, line_search_fn='strong_wolfe')
            for k in range(1,polish_steps+1):
                if expired():
                    reason = 'time_budget_exhausted'
                    break
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
    state = persist(reason)
    return state, history


def train_channel(model, out, fail_at=None, *, max_seconds=None, audit_every=None,
                  save_every_audit=False):
    c = model.cfg
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device=device).manual_seed(c.seed+101)
    def batch(gen,nt=None,ns=None):
        nt,ns=nt or c.time_batch,ns or c.space_batch
        t = torch.rand(nt, 1, generator=gen, dtype=dtype, device=device).square()*c.final_time
        xyz = torch.rand(nt, ns, c.dimension, generator=gen, dtype=dtype, device=device)
        return t, xyz*xyz.new_tensor(c.lengths)
    audit = batch(torch.Generator(device=device).manual_seed(19481),max(c.time_batch,7),max(c.space_batch,96))
    audit[0][0]=0.;audit[0][-1]=c.final_time
    polish = batch(torch.Generator(device=device).manual_seed(8516),max(c.time_batch,6),max(c.space_batch,96))
    polish[0][0]=0.;polish[0][-1]=c.final_time
    def objective(audit=False):
        return model.loss(*(audit_batch if audit else batch(generator)))
    audit_batch = audit
    return fit(model, objective, c.epochs, c.lr, out, decay=c.learning_rate_decay, fail_at=fail_at,
               polish_steps=c.lbfgs_steps,polish_function=lambda:model.loss(*polish),
               max_seconds=max_seconds,audit_every=audit_every,save_every_audit=save_every_audit)
