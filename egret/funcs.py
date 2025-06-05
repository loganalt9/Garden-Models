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

def load_atom_structure(atoms_file_path):
    from huggingface_hub import hf_hub_download
    import ase.io

    if not atoms_file_path:
        atoms_file_path = hf_hub_download(repo_id="lalt9/egret1", filename="example.xyz")
    
    try:
        atoms = ase.io.read(atoms_file_path, format="xyz")
    except FileNotFoundError:
        raise ValueError(f"Could not find atoms file at: {atoms_file_path}")
    
    return atoms

@app.function(image=image)
def predict_energies_and_forces(model="EGRET_1.model", atoms_file_path=None):
    from ase.calculators.calculator import all_changes

    atoms = load_atom_structure(atoms_file_path)
    calculator = load_model(model)
    
    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.results

@app.function(image=image)
def extract_equivariant_descriptor(model="EGRET_1.model", atoms_file_path=None):
    from ase.calculators.calculator import all_changes

    atoms = load_atom_structure(atoms_file_path)
    calculator = load_model(model)

    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.models[0](calculator._atoms_to_batch(atoms).to_dict())["node_feats"]

@app.function(image=image)
def extract_invariant_descriptor(model="EGRET_1.model", atoms_file_path=None):
    import torch

    equivariant = extract_equivariant_descriptor.remote(model, atoms_file_path)

    return torch.cat([equivariant[:, :192], equivariant[:, -192:]], dim=1)


@app.local_entrypoint()
def main():
    print(predict_energies_and_forces.remote())

    print(extract_equivariant_descriptor.remote().shape)

    print(extract_invariant_descriptor.remote().shape)
