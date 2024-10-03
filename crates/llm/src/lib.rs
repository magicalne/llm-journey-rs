use tikv_jemallocator::Jemalloc;

pub mod config;
pub mod deepseek2;
pub mod utils;
pub mod var_builder;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
