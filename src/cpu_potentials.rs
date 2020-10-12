use crate::cpu_sampler::{IntegratorImpl, LeapfrogInfoImpl, Potential, State, StateIdx};
use crate::integrator::AdaptationCollector;
use crate::nuts::SampleInfo;

type StaticDiagIntegrator = IntegratorImpl<StaticDiagPotential>;

pub(crate) struct EmptyCollector {}
impl AdaptationCollector<StaticDiagIntegrator> for EmptyCollector {
    fn inform_leapfrog(&mut self, _start: StateIdx, _info: &LeapfrogInfoImpl) {}
    fn inform_accept(
        &mut self,
        _old: StateIdx,
        _new: StateIdx,
        _info: &SampleInfo<StaticDiagIntegrator>,
    ) {
    }
    fn is_tuning(&self) -> bool {
        false
    }
}

pub(crate) struct StaticDiagPotential {
    diag: Box<[f64]>,
}

impl StaticDiagPotential {
    pub(crate) fn new(diag: Box<[f64]>) -> StaticDiagPotential {
        StaticDiagPotential { diag }
    }
}

impl Potential for StaticDiagPotential {
    type Integrator = IntegratorImpl<Self>;
    type Collector = EmptyCollector;

    fn update_self(&mut self, _collector: Self::Collector) {}
    fn update_state(&self, state: &mut State) {
        unimplemented!()
    }
    fn collector(&self) -> Self::Collector {
        EmptyCollector {}
    }
}
