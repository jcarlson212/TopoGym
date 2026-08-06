"""The algorithms themselves.

Kept apart from the surrounding machinery on purpose: the modules
beside this package -- the protocol, the evaluation harness, instance
construction, parallelism, reporting -- are shared by every baseline
and should not change when an algorithm is added. What lives here is
one module per algorithm, each a subclass of
:class:`~topogym.baselines.gridworld2dv1.protocol.Baseline`.

Resolve them by name through
:func:`topogym.baselines.gridworld2dv1.get_baseline` rather than
importing directly, so that listing the baselines never imports Ray or
torch.
"""
