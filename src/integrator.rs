use crate::nuts::SampleInfo;

pub trait LeapfrogInfo<D> {
    fn energy_error(&self) -> f64;
    fn divergence(&self) -> Option<D>;
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
    type DivergenceInfo: Copy;
    type LeapfrogInfo: LeapfrogInfo<Self::DivergenceInfo>;
    type StateIdx: Copy + Eq;

    fn initial_state(&self) -> Self::StateIdx;
    fn leapfrog(
        &mut self,
        start: Self::StateIdx,
        dir: Direction,
    ) -> (Self::StateIdx, Self::LeapfrogInfo);
    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool;
    fn free_state(&mut self, state: Self::StateIdx);
    fn accept(&mut self, state: Self::StateIdx, info: SampleInfo<Self::DivergenceInfo>);
}

pub struct Draw<I: Integrator, S> {
    state: I::StateIdx,
    info: S,
    tuning: bool,
}

pub trait Sampler {
    type Integrator: Integrator;
    type AdaptationCollector: AdaptationCollector<
        <Self::Integrator as Integrator>::StateIdx,
        <Self::Integrator as Integrator>::LeapfrogInfo,
        <Self::Integrator as Integrator>::DivergenceInfo,
    >;

    fn adapt(&mut self, collector: Self::AdaptationCollector);
    fn collector(&self) -> Self::AdaptationCollector;
    fn integrator(&mut self) -> &mut Self::Integrator;

    fn draw<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        state: <Self::Integrator as Integrator>::StateIdx,
        maxdepth: u64,
    ) -> Draw<Self::Integrator, SampleInfo<<Self::Integrator as Integrator>::DivergenceInfo>> {
        let mut collector = self.collector();
        let mut integrator: CollectingIntegrator<Self> = CollectingIntegrator {
            collector: &mut collector,
            integrator: self.integrator(),
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
    type DivergenceInfo = <S::Integrator as Integrator>::DivergenceInfo;
    type LeapfrogInfo = <S::Integrator as Integrator>::LeapfrogInfo;

    fn initial_state(&self) -> Self::StateIdx {
        self.initial_state
    }

    fn leapfrog(
        &mut self,
        start: Self::StateIdx,
        dir: Direction,
    ) -> (Self::StateIdx, Self::LeapfrogInfo) {
        let out = self.integrator.leapfrog(start, dir);
        self.collector.inform_leapfrog(out.0, &out.1);
        out
    }

    fn is_turning(&self, start: Self::StateIdx, end: Self::StateIdx) -> bool {
        self.integrator.is_turning(start, end)
    }

    fn free_state(&mut self, state: Self::StateIdx) {
        self.integrator.free_state(state)
    }

    fn accept(&mut self, state: Self::StateIdx, info: SampleInfo<Self::DivergenceInfo>) {
        self.collector
            .inform_accept(self.initial_state, state, &info);
        self.accept(state, info)
    }
}

pub trait AdaptationCollector<I, L, D> {
    fn inform_leapfrog(&mut self, start: I, info: &L);
    fn inform_accept(&mut self, old: I, new: I, info: &SampleInfo<D>);
    fn is_tuning(&self) -> bool;
}
