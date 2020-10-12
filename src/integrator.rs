use crate::nuts::SampleInfo;

pub trait DivergenceInfo { }

pub trait LeapfrogInfo {
    type DivergenceInfo: DivergenceInfo;

    fn energy_error(&self) -> f64;
    fn divergence(&mut self) -> Option<Self::DivergenceInfo>;
    fn diverging(&self) -> bool;
}

#[derive(Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl rand::distributions::Distribution<Direction> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.gen::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
}

pub trait Integrator {
    type LeapfrogInfo: LeapfrogInfo;
    type StateIdx: Copy + Eq;

    fn initial_state(&self) -> Self::StateIdx;
    fn leapfrog(
        &mut self,
        start: Self::StateIdx,
        dir: Direction,
    ) -> (Self::StateIdx, Self::LeapfrogInfo);
    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool;
    fn free_state(&mut self, state: Self::StateIdx);
    fn accept(&mut self, state: Self::StateIdx, info: SampleInfo<Self>);
    fn write_position(&self, state: Self::StateIdx, out: &mut [f64]);
}

pub struct Draw<S: Sampler + ?Sized> {
    state: <S::Integrator as Integrator>::StateIdx,
    info: SampleInfo<S::Integrator>,
    tuning: bool,
}

impl<S: Sampler> Draw<S> {
    fn position(&self, sampler: &S, out: &mut [f64]) {
        sampler.integrator().write_position(self.state, out);
    }

    fn info(&self) -> SampleInfo<S::Integrator> {
        unimplemented!()
    }

}

pub trait Sampler {
    type Integrator: Integrator;
    type AdaptationCollector: AdaptationCollector<Self::Integrator>;

    fn adapt(&mut self, collector: Self::AdaptationCollector);
    fn collector(&self) -> Self::AdaptationCollector;
    fn integrator_mut(&mut self) -> &mut Self::Integrator;
    fn integrator(&self) -> &Self::Integrator;

    fn draw<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        state: <Self::Integrator as Integrator>::StateIdx,
        maxdepth: u64,
    ) -> Draw<Self> {
        let mut collector = self.collector();
        let mut integrator: CollectingIntegrator<Self> = CollectingIntegrator {
            collector: &mut collector,
            integrator: self.integrator_mut(),
            initial_state: state,
        };
        let (idx, info) = crate::nuts::draw(rng, &mut integrator, maxdepth);
        Draw {
            state: idx,
            info,
            tuning: collector.is_tuning(),
        }
    }
}

struct CollectingIntegrator<'a, S: Sampler + ?Sized> {
    collector: &'a mut S::AdaptationCollector,
    integrator: &'a mut S::Integrator,
    initial_state: <S::Integrator as Integrator>::StateIdx,
}

impl<'a, S> Integrator for CollectingIntegrator<'a, S>
where
    S: Sampler + ?Sized,
{
    type StateIdx = <S::Integrator as Integrator>::StateIdx;
    type LeapfrogInfo = <S::Integrator as Integrator>::LeapfrogInfo;

    fn initial_state(&self) -> Self::StateIdx {
        self.initial_state
    }

    fn leapfrog(
        &mut self,
        start: Self::StateIdx,
        dir: Direction,
    ) -> (Self::StateIdx, Self::LeapfrogInfo) {
        let (state, info) = self.integrator.leapfrog(start, dir);
        self.collector.inform_leapfrog(state, &info);
        (state, info)
    }

    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool {
        self.integrator.is_turning(start, end)
    }

    fn free_state(&mut self, state: Self::StateIdx) {
        self.integrator.free_state(state)
    }

    fn accept(&mut self, state: Self::StateIdx, info: SampleInfo<Self>) {
        self.collector
            .inform_accept(self.initial_state, state, &info);
        self.accept(state, info)
    }

    fn write_position(&self, state: Self::StateIdx, out: &mut [f64]) {
        unimplemented!();
    }
}

pub trait AdaptationCollector<I: Integrator + ?Sized> {
    fn inform_leapfrog(&mut self, start: I::StateIdx, info: &I::LeapfrogInfo);
    fn inform_accept(&mut self, old: I::StateIdx, new: I::StateIdx, info: &SampleInfo<I>);
    fn is_tuning(&self) -> bool;
}
