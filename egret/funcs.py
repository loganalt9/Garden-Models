import modal
from pathlib import Path

app = modal.App("mace-egret")

model_path = Path("compiled_models")
atoms_path = Path("atoms")

image = modal.Image.debian_slim(python_version="3.12").apt_install("wget", "git").pip_install(
    "mace-torch==0.3.12", "ase", "torch", "huggingface_hub"
)
    
def load_model(model):
    from huggingface_hub import hf_hub_download
    from mace.calculators import mace_off

    raw_file_path = hf_hub_download(repo_id="lalt9/egret1", filename=model)
    calculator = mace_off(model=raw_file_path, default_dtype="float64")

    return calculator

def load_atom_structure(atoms_path):
    from huggingface_hub import hf_hub_download
    import ase.io
    import sys

    if not atoms_path:
        atoms_path = hf_hub_download(repo_id="lalt9/egret1", filename="example.xyz")
    
    try:
        atoms = ase.io.read(atoms_path, format="xyz")
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)
    
    return atoms



@app.function(image=image)
def predict_energies_and_forces(model="EGRET_1.model", atoms_path=None):
    from ase.calculators.calculator import all_changes

    atoms = load_atom_structure(atoms_path)
    calculator = load_model(model)
    
    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.results

@app.function(image=image)
def extract_equivariant_descriptor(model="EGRET_1.model", atoms_path=None):
    from ase.calculators.calculator import all_changes

    atoms = load_atom_structure(atoms_path)
    calculator = load_model(model)

    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.models[0](calculator._atoms_to_batch(atoms).to_dict())["node_feats"]

@app.function(image=image)
def extract_invariant_descriptor(model="EGRET_1.model", atoms_path=None):
    import torch

    equivariant = extract_equivariant_descriptor.remote(model, atoms_path)

    return torch.cat([equivariant[:, :192], equivariant[:, -192:]], dim=1)


@app.local_entrypoint()
def main():
    print(predict_energies_and_forces.remote())

    print(extract_equivariant_descriptor.remote())

    print(extract_invariant_descriptor.remote())