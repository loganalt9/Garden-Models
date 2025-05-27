import modal
from pathlib import Path

app = modal.App("mace-egret")

model_path = Path("compiled_models")
atoms_path = Path("atoms")


image = modal.Image.debian_slim().pip_install(
    "mace-torch==0.3.12", "ase", "torch"
).add_local_dir(
    model_path, remote_path="/models"
).add_local_dir(
    atoms_path, remote_path="/atoms" # For testing purposes
)



@app.function(image=image)
def predict_energies_and_forces(atoms_path, model="EGRET_1.model"):
    import ase.io
    from ase.calculators.calculator import all_changes
    from mace.calculators import mace_off

    atoms = ase.io.read(atoms_path, format="xyz")
    calculator = mace_off(model=f'/models/{model}', default_dtype="float64")
    
    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.results

@app.function(image=image)
def extract_equivariant_descriptor(atoms_path, model="EGRET_1.model"):
    import ase.io
    from mace.calculators import mace_off
    from ase.calculators.calculator import all_changes

    atoms = ase.io.read(atoms_path, format="xyz")
    calculator = mace_off(model=f'/models/{model}', default_dtype="float64")
    calculator.calculate(atoms, ["energy", "forces"], all_changes)

    return calculator.models[0](calculator._atoms_to_batch(atoms).to_dict())["node_feats"]

@app.function(image=image)
def extract_invariant_descriptor(atoms_path, model="EGRET_1.model"):
    import torch

    equivariant = extract_equivariant_descriptor.remote(atoms_path, model)

    return torch.cat([equivariant[:, :192], equivariant[:, -192:]], dim=1)


@app.local_entrypoint()
def main():
    atoms_path = "/atoms/example.xyz"

    print(predict_energies_and_forces.remote(atoms_path))

    print(extract_equivariant_descriptor.remote(atoms_path))

    print(extract_invariant_descriptor.remote(atoms_path))