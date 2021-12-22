use thiserror::Error;

use std::marker::PhantomData;

use crate::math::logaddexp;

#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned unrecoverable error")]
    LogpFailure(Box<dyn std::error::Error + Send>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

pub trait DivergenceInfo: std::fmt::Debug + Send {
    /// The position in parameter space where the diverging leapfrog started
    fn start_location(&self) -> Option<&[f64]> {
        None
    }

    /// The position in parameter space where the diverging leapfrog ended
    fn end_location(&self) -> Option<&[f64]> {
        None
    }

    /// The difference between the energy at the initial location of the trajectory and
    /// the energy at the end of the diverging leapfrog step.
    fn energy_error(&self) -> Option<f64> {
        None
    }
    /// The index of the end location of the diverging leapfrog.
    fn end_idx_in_trajectory(&self) -> Option<i64> {
        None
    }
}

#[derive(Copy, Clone)]
pub(crate) enum Direction {
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

/// Callbacks for various events during a Nuts sampling step.
pub(crate) trait Collector {
    type State: State;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        _end: &Self::State,
        _divergence_info: Option<&dyn DivergenceInfo>,
    ) {
    }
    fn register_draw(&mut self, _state: &Self::State, _info: &SampleInfo) {}
    fn register_init(&mut self, _state: &Self::State, _options: &NutsOptions) {}
}

pub trait LogpError: std::error::Error {
    fn is_recoverable(&self) -> bool;
}

pub(crate) trait Potential {
    type State: State;
    type DivergenceInfo: DivergenceInfo + 'static;
    type LogpError: LogpError;
    type Stats: Copy + Send;

    fn leapfrog<C: Collector<State = Self::State>>(
        &mut self,
        pool: &mut <Self::State as State>::Pool,
        start: &Self::State,
        dir: Direction,
        initial_energy: f64,
        collector: &mut C,
    ) -> Result<std::result::Result<Self::State, Self::DivergenceInfo>>;

    fn init_state(
        &mut self,
        pool: &mut <Self::State as State>::Pool,
        init: &[f64],
    ) -> Result<Self::State>;

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R);

    fn current_stats(&self) -> Self::Stats;

    fn new_empty_state(&mut self, pool: &mut <Self::State as State>::Pool) -> Self::State;

    fn new_pool(&mut self, capacity: usize) -> <Self::State as State>::Pool;

    fn dim(&self) -> usize;
}

pub(crate) trait State: Clone {
    type Pool;

    fn write_position(&self, out: &mut [f64]);
    fn is_turning(&self, other: &Self) -> bool;
    fn energy(&self) -> f64;
    fn potential_energy(&self) -> f64;
    fn index_in_trajectory(&self) -> i64;
    fn into_init_point(&mut self);

    fn log_acceptance_probability(&self, initial_energy: f64) -> f64 {
        (initial_energy - self.energy()).min(0.)
    }
}

#[derive(Debug)]
pub(crate) struct SampleInfo {
    pub depth: u64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub reached_maxdepth: bool,
}

struct NutsTree<P: Potential, C: Collector<State = P::State>> {
    left: P::State,
    right: P::State,
    draw: P::State,
    log_size: f64,
    depth: u64,
    initial_energy: f64,
    is_main: bool,
    collector: PhantomData<C>,
}

enum ExtendResult<P: Potential, C: Collector<State = P::State>> {
    Ok(NutsTree<P, C>),
    Err(NutsError),
    Turning(NutsTree<P, C>),
    Diverging(NutsTree<P, C>, P::DivergenceInfo),
}

impl<P: Potential, C: Collector<State = P::State>> NutsTree<P, C> {
    fn new(state: P::State) -> NutsTree<P, C> {
        let initial_energy = state.energy();
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: 0.,
            initial_energy,
            is_main: true,
            collector: PhantomData,
        }
    }

    #[inline]
    fn extend<R>(
        mut self,
        pool: &mut <P::State as State>::Pool,
        rng: &mut R,
        potential: &mut P,
        direction: Direction,
        options: &NutsOptions,
        collector: &mut C,
    ) -> ExtendResult<P, C>
    where
        P: Potential,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(pool, potential, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(pool, rng, potential, direction, options, collector) {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
                Err(error) => {
                    return Err(error);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (&self.left, &other.right),
            Direction::Backward => (&other.left, &self.right),
        };

        let mut turning = first.is_turning(last);
        if self.depth > 0 {
            if !turning {
                turning = self.right.is_turning(&other.right);
            }
            if !turning {
                turning = self.left.is_turning(&other.left);
            }
        }

        self.merge_into(other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    #[inline]
    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        other: NutsTree<P, C>,
        rng: &mut R,
        direction: Direction,
    ) {
        assert!(self.depth == other.depth);
        assert!(self.left.index_in_trajectory() <= self.right.index_in_trajectory());
        match direction {
            Direction::Forward => {
                self.right = other.right;
            }
            Direction::Backward => {
                self.left = other.left;
            }
        }
        let log_size = logaddexp(self.log_size, other.log_size);

        let self_log_size = if self.is_main {
            assert!(self.left.index_in_trajectory() <= 0);
            assert!(self.right.index_in_trajectory() >= 0);
            self.log_size
        } else {
            log_size
        };

        if other.log_size >= self_log_size {
            self.draw = other.draw;
        } else if rng.gen_bool((other.log_size - self_log_size).exp()) {
            self.draw = other.draw;
        }

        self.depth += 1;
        self.log_size = log_size;
    }

    #[inline]
    fn single_step(
        &self,
        pool: &mut <P::State as State>::Pool,
        potential: &mut P,
        direction: Direction,
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<P, C>, P::DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match potential.leapfrog(pool, start, direction, self.initial_energy, collector) {
            Ok(Ok(end)) => end,
            Ok(Err(info)) => return Ok(Err(info)),
            Err(error) => return Err(error),
        };

        let log_size = self.initial_energy - end.energy();
        Ok(Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size,
            initial_energy: self.initial_energy,
            is_main: false,
            collector: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<P::DivergenceInfo>) -> SampleInfo {
        let info: Option<Box<dyn DivergenceInfo>> = match divergence_info {
            Some(info) => Some(Box::new(info)),
            None => None,
        };
        SampleInfo {
            depth: self.depth,
            divergence_info: info,
            reached_maxdepth: maxdepth,
        }
    }
}

pub(crate) struct NutsOptions {
    pub maxdepth: u64,
}

pub(crate) fn draw<P, R, C>(
    pool: &mut <P::State as State>::Pool,
    init: &mut P::State,
    rng: &mut R,
    potential: &mut P,
    options: &NutsOptions,
    collector: &mut C,
) -> Result<(P::State, SampleInfo)>
where
    P: Potential,
    R: rand::Rng + ?Sized,
    C: Collector<State = P::State>,
{
    let mut init = init;
    potential.randomize_momentum(&mut init, rng);
    init.into_init_point();
    collector.register_init(&init, options);

    let mut tree = NutsTree::new(init.clone());
    while tree.depth < options.maxdepth {
        let direction: Direction = rng.gen();
        tree = match tree.extend(pool, rng, potential, direction, options, collector) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(tree) => {
                let info = tree.info(false, None);
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
    let info = tree.info(true, None);
    Ok((tree.draw, info))
}

#[derive(Debug)]
pub(crate) struct SampleStats<P: Potential, S: AdaptStrategy> {
    //pub step_size: f64,
    //pub step_size_bar: f64,
    pub depth: u64,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: i64,
    pub logp: f64,
    pub energy: f64,
    //pub mean_acceptance_rate: f64,
    pub divergence_info: Option<Box<dyn DivergenceInfo>>,
    pub chain: u64,
    pub draw: u64,
    //pub tree_size: u64,
    //pub first_diag_mass_matrix: f64,
    pub potential_stats: P::Stats,
    pub strategy_stats: S::Stats,
}

pub(crate) trait Sampler {
    type Potential: Potential;
    type AdaptStrategy: AdaptStrategy;

    fn set_position(&mut self, position: &[f64]) -> Result<()>;
    fn draw(
        &mut self,
    ) -> Result<(
        Box<[f64]>,
        SampleStats<Self::Potential, Self::AdaptStrategy>,
    )>;
    fn dim(&self) -> usize;
}

pub(crate) struct NutsSampler<P, R, S>
where
    P: Potential,
    R: rand::Rng,
    S: AdaptStrategy<Potential = P>,
{
    pool: <P::State as State>::Pool,
    potential: P,
    collector: S::Collector,
    options: NutsOptions,
    rng: R,
    init: P::State,
    chain: u64,
    draw_count: u64,
    strategy: S,
}

impl<P, R, S> NutsSampler<P, R, S>
where
    P: Potential,
    R: rand::Rng,
    S: AdaptStrategy<Potential = P>,
{
    pub(crate) fn new(
        mut potential: P,
        strategy: S,
        options: NutsOptions,
        rng: R,
        chain: u64,
    ) -> Self {
        let pool_size: usize = options.maxdepth.checked_mul(2).unwrap().try_into().unwrap();
        let mut pool = potential.new_pool(pool_size);
        let init = potential.new_empty_state(&mut pool);
        let collector = strategy.new_collector();
        NutsSampler {
            pool,
            potential,
            collector,
            options,
            rng,
            init,
            chain,
            draw_count: 0,
            strategy,
        }
    }
}

pub(crate) trait AdaptStrategy {
    type Potential: Potential;
    type Collector: Collector<State = <Self::Potential as Potential>::State>;
    type Stats: Copy + Send;
    type Options: Copy + Send + Default;

    fn new(options: Self::Options, num_tune: u64, dim: usize) -> Self;

    fn adapt(&mut self, potential: &mut Self::Potential, draw: u64, collector: &Self::Collector);

    fn new_collector(&self) -> Self::Collector;

    fn current_stats(&self, collector: &Self::Collector) -> Self::Stats;
}

impl<P, R, S> Sampler for NutsSampler<P, R, S>
where
    P: Potential,
    R: rand::Rng,
    S: AdaptStrategy<Potential = P>,
{
    type Potential = P;
    type AdaptStrategy = S;

    fn set_position(&mut self, position: &[f64]) -> Result<()> {
        self.potential
            .init_state(&mut self.pool, position)
            .map(|_| ())
    }

    fn draw(
        &mut self,
    ) -> Result<(
        Box<[f64]>,
        SampleStats<Self::Potential, Self::AdaptStrategy>,
    )> {
        let (state, info) = draw(
            &mut self.pool,
            &mut self.init,
            &mut self.rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        )?;
        let position: Box<[f64]> = vec![0f64; self.potential.dim()].into();
        let stats = SampleStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            idx_in_trajectory: state.index_in_trajectory(),
            logp: -state.potential_energy(),
            energy: state.energy(),
            divergence_info: info.divergence_info,
            chain: self.chain,
            draw: self.draw_count,
            potential_stats: self.potential.current_stats(),
            strategy_stats: self.strategy.current_stats(&self.collector),
            /*  TODO
            tree_size: info.
            step_size: self.options.step_size,
            step_size_bar: self.options.step_size,
            divergence_info: info.divergence_info,
            idx_in_trajectory: self.state.idx_in_trajectory,
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            logp: -self.state.potential_energy,
            mean_acceptance_rate: self.collector.stats().mean_acceptance_rate,
            chain: self.chain,
            draw: self.draw_count,
            tree_size: self.collector.acceptance_rate.mean.count,
            first_diag_mass_matrix: self.potential.mass_matrix_mut().current.variance[0],
            */
        };
        self.strategy
            .adapt(&mut self.potential, self.draw_count, &mut self.collector);
        self.init = state;
        self.draw_count += 1;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.potential.dim()
    }
}
