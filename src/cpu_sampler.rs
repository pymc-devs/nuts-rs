use crate::integrator::{
    AdaptationCollector, Direction, DivergenceInfo, Integrator, LeapfrogInfo, Sampler,
};
use crate::nuts::SampleInfo;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, LinkedList};
use typed_arena::Arena;

pub(crate) type StateIdx = typed_arena::Arena::

pub(crate) struct State {
    free_link: LinkedListLink,
    p: Box<[f64]>,
    q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    potential_energy: f64,
}

fn empty_box(size: usize) -> Box<[f64]> {
    vec![0.; size].into()
}

impl State {
    fn energy(&self) -> f64 {
        self.kinetic_energy + self.potential_energy
    }

    fn new(position: &[f64], gradient: &[f64]) -> State {
        let dim = position.len();
        let mut p = empty_box(dim);
        p.copy_from_slice(position);
        State {
            p: position.into(),
            q: empty_box(dim),
            v: empty_box(dim),
            p_sum: empty_box(dim),
            grad: gradient.into(),
            idx_in_trajectory: 0,
            kinetic_energy: 0.,
            potential_energy: 0.,
        }
    }
}

pub(crate) trait Potential<F: LogpFunc>: Sized {
    type Collector: AdaptationCollector<IntegratorImpl<Self, F>>;

    fn update_state(&self, integrator: &IntegratorImpl<Self, F>, state: &mut State);
    fn adapt(&mut self, integrator: &IntegratorImpl<Self, F>, collector: Self::Collector);
    fn collector(&self) -> Self::Collector;
}

pub(crate) struct IntegratorImpl<'a, P: Potential<F>, F: LogpFunc> {
    free_states: LinkedList<ValueAdapter<'a>>,
    initial_state: StateIdx,
    potential: Option<P>,
    logp: F,
}

impl<P: Potential<F>, F: LogpFunc> IntegratorImpl<P, F> {
    fn new(
        logp: F,
        capacity: usize,
        position: &[f64],
        gradient: &[f64],
        potential: P,
    ) -> IntegratorImpl<P, F> {
        let mut arena = generational_arena::Arena::with_capacity(capacity);
        let state = State::new(position, gradient);
        let state_idx = arena.insert(state);
        IntegratorImpl {
            states: arena,
            initial_state: state_idx,
            potential: Some(potential),
            logp,
        }
    }

    fn adapt_potential(&mut self, collector: P::Collector) {
        let mut potential = self.potential.take().unwrap();
        potential.adapt(&*self, collector);
        self.potential = Some(potential);
    }
}

pub(crate) struct DivergenceInfoImpl {}

impl DivergenceInfo for DivergenceInfoImpl {}

pub(crate) struct LeapfrogInfoImpl {
    energy_error: f64,
    divergence: Option<DivergenceInfoImpl>,
}

impl LeapfrogInfo for LeapfrogInfoImpl {
    type DivergenceInfo = DivergenceInfoImpl;

    fn energy_error(&self) -> f64 {
        self.energy_error
    }

    fn divergence(&mut self) -> Option<DivergenceInfoImpl> {
        self.divergence.take()
    }

    fn diverging(&self) -> bool {
        self.divergence.is_some()
    }
}

struct SamplerImpl<P: Potential<F>, F: LogpFunc> {
    integrator: IntegratorImpl<P, F>,
}

impl<P: Potential<F>, F: LogpFunc> SamplerImpl<P, F> {
    fn new(integrator: IntegratorImpl<P, F>) -> SamplerImpl<P, F> {
        SamplerImpl { integrator }
    }
}

impl<P: Potential<F>, F: LogpFunc> Sampler for SamplerImpl<P, F>
where
    P::Collector: AdaptationCollector<IntegratorImpl<P, F>>,
{
    type Integrator = IntegratorImpl<P, F>;
    type AdaptationCollector = P::Collector;

    fn integrator_mut(&mut self) -> &mut Self::Integrator {
        &mut self.integrator
    }

    fn integrator(&self) -> &Self::Integrator {
        &self.integrator
    }

    fn collector(&self) -> Self::AdaptationCollector {
        self.integrator.potential.as_ref().unwrap().collector()
    }

    fn adapt(&mut self, collector: Self::AdaptationCollector) {
        self.integrator.adapt_potential(collector);
    }
}

impl<P: Potential<F>, F: LogpFunc> Integrator for IntegratorImpl<P, F> {
    type LeapfrogInfo = LeapfrogInfoImpl;
    type StateIdx = StateIdx;

    fn initial_state(&self) -> Self::StateIdx {
        let state = self.initial_state;
        assert!(self.states.contains(state));
        state
    }

    fn leapfrog(
        &mut self,
        start: Self::StateIdx,
        dir: Direction,
    ) -> (Self::StateIdx, Self::LeapfrogInfo) {
        unimplemented!()
    }

    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool {
        use crate::math::scalar_prods_of_diff;

        let start = &self.states[start];
        let end = &self.states[end];

        let (start, end) = if start.idx_in_trajectory < end.idx_in_trajectory {
            (start, end)
        } else {
            (end, start)
        };

        let (a, b) = scalar_prods_of_diff(&end.p_sum, &start.p_sum, &end.v, &start.v);
        (a < 0.) | (b < 0.)
    }

    fn free_state(&mut self, idx: Self::StateIdx) {
        self.states.remove(idx).expect("Double free");
    }

    fn accept(&mut self, state: Self::StateIdx, info_: &SampleInfo<Self>) {
        assert!(self.states.contains(state));
        assert!(self.states.len() == 1);
    }

    fn write_position(&self, state: Self::StateIdx, out: &mut [f64]) {
        let state = self.states.get(state).expect("Use after free");
        out.copy_from_slice(&state.p);
    }
}

pub trait LogpFunc {
    fn dim(&self) -> usize;
    fn logp_dlogp(&self, point: &[f64], out: &mut [f64]) -> f64;
}

fn sampler<P, F>(potential: P, init: Box<[f64]>, logp: F) -> impl Sampler
where
    P: Potential<F>,
    F: LogpFunc,
    P::Collector: AdaptationCollector<IntegratorImpl<P, F>>,
{
    let dim = logp.dim();
    if dim != init.len() {
        panic!("Shape mismatch");
    }
    let capacity = 1024;
    let mut grad = empty_box(dim);
    logp.logp_dlogp(&init, &mut grad);
    let integrator = IntegratorImpl::new(logp, capacity, &init, &grad, potential);
    SamplerImpl::new(integrator)
}

pub fn sampler_staticdiag<F: LogpFunc>(
    init: Box<[f64]>,
    logp: F,
    diag: Box<[f64]>,
) -> impl Sampler {
    use crate::cpu_potentials::StaticDiagPotential;
    let potential = StaticDiagPotential::new(diag);
    sampler(potential, init, logp)
}
