use crate::integrator::{
    AdaptationCollector, Direction, DivergenceInfo, Integrator, LeapfrogInfo, Sampler,
};
use crate::nuts::SampleInfo;

pub(crate) type StateIdx = generational_arena::Index;

pub(crate) struct State {
    p: Box<[f64]>,
    q: Box<[f64]>,
    v: Box<[f64]>,
    p_sum: Box<[f64]>,
    grad: Box<[f64]>,
    idx_in_trajectory: i64,
    kinetic_energy: f64,
    potential_energy: f64,
    used: bool,
}

impl State {
    fn energy(&self) -> f64 {
        self.kinetic_energy + self.potential_energy
    }

    fn new(position: &[f64]) -> State {
        unimplemented!()
    }
}

pub(crate) trait Potential {
    type Integrator: Integrator;
    type Collector: AdaptationCollector<Self::Integrator>;

    fn update_state(&self, state: &mut State);
    fn update_self(&mut self, collector: Self::Collector);
    fn collector(&self) -> Self::Collector;
}

pub(crate) struct IntegratorImpl<P: Potential> {
    states: generational_arena::Arena<State>,
    initial_state: StateIdx,
    potential: P,
}

impl<P: Potential> IntegratorImpl<P> {
    fn new(capacity: usize, position: &[f64], potential: P) -> IntegratorImpl<P> {
        let mut arena = generational_arena::Arena::with_capacity(capacity);
        let state = State::new(position);
        let state_idx = arena.insert(state);
        IntegratorImpl {
            states: arena,
            initial_state: state_idx,
            potential,
        }
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

struct SamplerImpl<P: Potential> {
    integrator: IntegratorImpl<P>,
}

impl<P: Potential> SamplerImpl<P> {
    fn new(integrator: IntegratorImpl<P>) -> SamplerImpl<P> {
        SamplerImpl { integrator }
    }
}

impl<P: Potential> Sampler for SamplerImpl<P>
where
    P::Collector: AdaptationCollector<IntegratorImpl<P>>,
{
    type Integrator = IntegratorImpl<P>;
    type AdaptationCollector = P::Collector;

    fn integrator_mut(&mut self) -> &mut Self::Integrator {
        &mut self.integrator
    }

    fn integrator(&self) -> &Self::Integrator {
        &self.integrator
    }

    fn collector(&self) -> Self::AdaptationCollector {
        self.integrator.potential.collector()
    }

    fn adapt(&mut self, collector: Self::AdaptationCollector) {
        self.integrator.potential.update_self(collector);
    }
}

impl<P: Potential> Integrator for IntegratorImpl<P> {
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
        unimplemented!();
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

    fn accept(&mut self, _state: Self::StateIdx, info_: SampleInfo<Self>) {
        unimplemented!()
    }

    fn write_position(&self, state: Self::StateIdx, out: &mut [f64]) {
        let state = self.states.get(state).expect("Use after free");
        out.copy_from_slice(&state.p);
    }
}

pub trait LogpFunc {
    fn dim(&self) -> usize;
    fn logp_dlogp(point: &[f64], out: &mut [f64]) -> f64;
}

fn sampler<P, F>(potential: P, init: Box<[f64]>, logp: F) -> impl Sampler
where
    P: Potential,
    F: LogpFunc,
    P::Collector: AdaptationCollector<IntegratorImpl<P>>,
{
    let dim = logp.dim();
    if dim != init.len() {
        panic!("Shape mismatch");
    }
    let capacity = 1024;
    let integrator = IntegratorImpl::new(capacity, &init, potential);
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
