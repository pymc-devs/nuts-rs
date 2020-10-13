use crate::cpu_sampler::{IntegratorImpl, LeapfrogInfoImpl, LogpFunc, Potential, State, StateIdx};
use crate::integrator::AdaptationCollector;
use crate::nuts::SampleInfo;

type StaticDiagIntegrator<F: LogpFunc> = IntegratorImpl<StaticDiagPotential<F>, F>;

pub(crate) struct EmptyCollector {
    accept_sum: f64,
}

impl<F: LogpFunc> AdaptationCollector<StaticDiagIntegrator<F>> for EmptyCollector {
    fn inform_leapfrog(&mut self, _integrator: &StaticDiagIntegrator<F>, _state: StateIdx, _info: &LeapfrogInfoImpl) {

    }
    fn inform_accept(
        &mut self,
        _integrator: &StaticDiagIntegrator<F>,
        _old: StateIdx,
        _new: StateIdx,
        _info: &SampleInfo<StaticDiagIntegrator<F>>,
    ) {
    }
    fn is_tuning(&self) -> bool {
        false
    }
}

pub(crate) struct StaticDiagPotential<F: LogpFunc> {
    diag: Box<[f64]>,
    _marker: std::marker::PhantomData<F>
}

impl<F: LogpFunc> StaticDiagPotential<F> {
    pub(crate) fn new(diag: Box<[f64]>) -> StaticDiagPotential<F> {
        StaticDiagPotential { diag, _marker: std::marker::PhantomData }
    }
}

impl<F: LogpFunc> Potential<F> for StaticDiagPotential<F> {
    type Collector = EmptyCollector;

    fn adapt(&mut self, integrator: &IntegratorImpl<Self, F>, _collector: Self::Collector) {
        unimplemented!()
    }
    fn update_state(&self, integrator: &IntegratorImpl<Self, F>, state: &mut State) {
        unimplemented!()
    }
    fn collector(&self) -> Self::Collector {
        EmptyCollector { accept_sum: 0. }
    }
}
