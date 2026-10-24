import pytest
import torch

from zentrainer.harness.objective import Objective, sample_percents, sigmas_for, timestep_weight
from zentrainer.spec import ObjectiveSpec, SpecError, TrainSpec


def _noise(spec, n=20_000):
    """Noise fractions drawn (1 = full noise)."""
    return 1.0 - sample_percents(spec, n, torch.Generator().manual_seed(0))


def test_sigmoid_is_ai_toolkits_sigmoid():
    # AI Toolkit's "sigmoid": noise = sigmoid(N(0, 1)), unshifted
    t = _noise(ObjectiveSpec(timesteps="logit_normal"))
    ref = torch.sigmoid(torch.randn(20_000, generator=torch.Generator().manual_seed(1)))
    for q in (0.1, 0.5, 0.9):
        assert abs(t.quantile(q) - ref.quantile(q)) < 0.01


def test_high_noise_bias_is_ai_toolkits_cubic():
    # "content": index u³·N into the timesteps sorted from most noise down = the 1 - u³ quantile
    t = _noise(ObjectiveSpec(bias="high_noise"))
    assert abs(t.median() - (1 - 0.5 ** 3)) < 0.01  # half the steps in the noisiest eighth
    lo = _noise(ObjectiveSpec(bias="low_noise"))
    assert abs(lo.median() - 0.5 ** 3) < 0.01
    # with sigmoid: the same quantiles of the logit-normal
    s = _noise(ObjectiveSpec(timesteps="logit_normal", bias="high_noise"))
    assert abs(s.median() - torch.sigmoid(torch.special.ndtri(torch.tensor(0.875)))) < 0.01
    assert s.median() > _noise(ObjectiveSpec(timesteps="logit_normal")).median() + 0.2


def test_spread_covers_the_schedule_evenly_and_resumes_in_place():
    spec = ObjectiveSpec(spread=True)
    first = 1.0 - sample_percents(spec, 10, start=0, seed=7)
    hist = torch.histc(first, bins=10, min=0, max=1)
    assert hist.max() <= 2  # ten steps land in (nearly) ten different tenths
    again = torch.cat([sample_percents(spec, 4, start=0, seed=7), sample_percents(spec, 6, start=4, seed=7)])
    assert torch.allclose(1.0 - again, first)  # drawn in parts, as after a resume: the same levels
    assert not torch.allclose(sample_percents(spec, 10, start=0, seed=8), 1.0 - first)


def test_flex_weighting_leans_to_heavy_noise():
    spec = ObjectiveSpec(weighting="flex")
    w = timestep_weight(spec, torch.tensor([1.0, 0.85, 0.5, 0.02]))
    assert w[1] > 1.45 and w[3] < 0.6 and 0.9 < w[2] < 1.1
    assert timestep_weight(ObjectiveSpec(), torch.tensor([0.5])) is None
    x0 = torch.zeros(2, 4, 2, 2)
    pred = torch.ones(2, 4, 2, 2)
    sigma = torch.tensor([0.85, 0.02])
    o = Objective(spec)
    plain, weighted = o.loss(x0, x0, pred, sigma), o.train_loss(x0, x0, pred, sigma)
    assert weighted[0] > plain[0] and weighted[1] < plain[1]


def test_a_steps_numbers_are_read_together_at_its_end():
    # the noise chart's draws stay unread through the step (reading them stalls the GPU), then
    # settle hands back the loss and grad norm with them, in the order each micro-batch drew
    o = Objective(ObjectiveSpec(target="x0"))  # a plain MSE against x0: the losses are scale²
    o.start_step()
    x0 = torch.zeros(2, 4, 2, 2)
    for scale, sigma in ((1.0, torch.tensor([0.9, 0.2])), (2.0, torch.tensor([0.5, 0.1]))):
        o.train_loss(x0, x0, torch.full_like(x0, scale), sigma, percents=1 - sigma)
    assert o.drawn == []
    loss, norm = o.settle(torch.tensor(3.5), torch.tensor(0.25))
    assert (loss, norm) == (3.5, 0.25)
    assert [(round(p, 2), round(s, 2), v) for p, s, v in o.drawn] == \
        [(0.1, 0.9, 1.0), (0.8, 0.2, 1.0), (0.5, 0.5, 4.0), (0.9, 0.1, 4.0)]
    o.start_step()
    assert o.drawn == [] and o.settle(torch.tensor(1.0)) == [1.0]


def test_bias_reaches_sigmas_through_a_shift():
    p = sample_percents(ObjectiveSpec(bias="high_noise"), 1000, torch.Generator().manual_seed(0))
    assert sigmas_for(None, p, shift=1.0).median() > 0.85


def test_bias_and_weighting_are_checked():
    base = {"name": "x", "dataset": {"folder": "x"}, "model": {"ckpt_name": "m.safetensors"}}
    for bad in ({"bias": "content"}, {"weighting": "bell"}, {"bias_power": 0.5}):
        with pytest.raises(SpecError):
            TrainSpec.from_dict({**base, "objective": bad}).validate()


def test_balance_evens_out_the_noise_levels_it_measures():
    from zentrainer.harness.objective import Balance

    b = Balance()
    hard, easy = torch.tensor([0.05]), torch.tensor([0.95])  # where in the schedule
    first = b(hard, torch.tensor([4.0]))
    assert first.item() == 1.0  # nothing measured yet: the plain loss
    for _ in range(Balance.WARM):
        b(hard, torch.tensor([4.0]))
        b(easy, torch.tensor([1.0]))
    w_hard, w_easy = b(hard, torch.tensor([4.0])).item(), b(easy, torch.tensor([1.0])).item()
    assert w_hard * 4.0 == pytest.approx(w_easy * 1.0, rel=0.05)  # each pulls about equally
    again = Balance()
    again.load_state_dict(b.state_dict())
    assert again(easy, torch.tensor([1.0])).item() == pytest.approx(b(easy, torch.tensor([1.0])).item(), rel=0.05)


def test_a_scheduler_trains_on_exactly_its_steps():
    schedule = torch.tensor([1.0, 0.8, 0.5, 0.2])
    p = sample_percents(ObjectiveSpec(timesteps="scheduler"), 4000, torch.Generator().manual_seed(0))
    s = sigmas_for(None, p, schedule=schedule)
    assert set(s.tolist()) == set(schedule.tolist())
    assert all(abs((s == v).float().mean() - 0.25) < 0.03 for v in schedule)  # each step as often
    hi = sigmas_for(None, sample_percents(ObjectiveSpec(timesteps="scheduler", bias="high_noise"), 4000,
                                          torch.Generator().manual_seed(0)), schedule=schedule)
    assert (hi == 1.0).float().mean() > 0.5  # a bias leans it too
