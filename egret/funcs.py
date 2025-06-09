import modal

app = modal.App("Egret1")

def download_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo_id="lalt9/egret1")

    for file in files:
        if file.startswith("EGRET"):
            hf_hub_download(repo_id="lalt9/egret1", filename=file, local_dir="models")

image = modal.Image.debian_slim(python_version="3.12").apt_install("wget", "git").pip_install(
    "mace-torch==0.3.12", "ase", "torch", "huggingface_hub"
).run_function(
    download_model
)

def load_model(model):
    from mace.calculators import mace_off

    raw_file_path = f"models/{model}"

    calculator = mace_off(model=raw_file_path, default_dtype="float64")

    return calculator

def _validate_atoms(atoms_dict):
    from ase import Atoms

    if isinstance(atoms_dict, dict):
        valid_keys = {
            "symbols", "positions", "numbers", "tags", "momenta",
            "masses", "magmoms", "charges", "scaled_positions",
            "cell", "pbc", "celldisp", "constraint", "calculator",
            "info", "velocities"
        }

        filtered = {k: v for k, v in atoms_dict.items() if k in valid_keys}

        atoms = Atoms(**filtered)
        
        return atoms
    elif isinstance(atoms_dict, Atoms):
        return atoms_dict
    else:
        raise TypeError("Expected an Atoms object or a dict")

@app.function(image=image)
def predict_energies_and_forces(atoms, model="EGRET_1.model"):
    from ase.calculators.calculator import all_changes

    calculator = load_model(model)
    atoms = _validate_atoms(atoms)

    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.results

@app.function(image=image)
def extract_equivariant_descriptor(atoms, model="EGRET_1.model"):
    from ase.calculators.calculator import all_changes

    calculator = load_model(model)
    atoms = _validate_atoms(atoms)

    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.models[0](calculator._atoms_to_batch(atoms).to_dict())["node_feats"]

@app.function(image=image)
def extract_invariant_descriptor(atoms, model="EGRET_1.model"):
    import torch

    atoms = _validate_atoms(atoms)
    equivariant = extract_equivariant_descriptor.remote(atoms, model)

    return torch.cat([equivariant[:, :192], equivariant[:, -192:]], dim=1)


@app.local_entrypoint()
def main():
    structure = {
        "symbols": ["O", "H", "H"],
        "positions": [[0,0,0], [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    }

    print(predict_energies_and_forces.remote(structure))

    print(extract_equivariant_descriptor.remote(structure))

    print(extract_invariant_descriptor.remote(structure))
