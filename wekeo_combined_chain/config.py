from wekeo_combined_chain.hygeos_core import env
output_dir = env.getdir("OUTPUT_DIR")

if not output_dir.exists():
    raise FileNotFoundError(f"Output directory {output_dir} does not exist. Please create it or check your environment configuration.")

gridded_combined_dir = output_dir / "gridded_combined"
gridded_combined_dir.mkdir(parents=False, exist_ok=True)