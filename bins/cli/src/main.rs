use clap::{Parser, Subcommand};
use llm::config::Config;
use std::fs;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run { file_path: String },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run { file_path } => {
            let toml_content = match fs::read_to_string(file_path) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("Failed to read file: {}", e);
                    return;
                }
            };

            let toml_value: Config = match toml::from_str::<Config>(&toml_content) {
                Ok(value) => value,
                Err(e) => {
                    eprintln!("Failed to parse TOML: {}", e);
                    return;
                }
            };

            println!("Parsed TOML: {:?}", toml_value);
        }
    }
}
