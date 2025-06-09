import modal

def download_models():
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="gomesgroup/simg", filename="lp_pred_model.ckpt", local_dir="models")
    hf_hub_download(repo_id="gomesgroup/simg", filename="nbo_pred_model.ckpt", local_dir="models")

def stage_all_files():
    import shutil

    shutil.move("/tmpsimg/simg", "/simg")
    #shutil.move("/tmpsimg/simg/data.py", "/simg/data.py")
    #shutil.move("/tmpsimg/simg/model_utils.py", "/simg/model_utils.py")
    #shutil.move("/tmpsimg/simg/utils.py", "/simg/utils.py")


image = modal.Image.debian_slim().apt_install([
    "git"
]).pip_install([
    "torch",
    "huggingface_hub",
    "numpy",
    "joblib",
    "torch_geometric",
    "scipy",
    "pytorch-lightning"
]).run_commands(
    "git clone https://github.com/gomesgroup/simg.git --depth 1 --single-branch /tmpsimg"
).run_function(
    stage_all_files
)

app = modal.App("simg")

# @app.function(image=image)
# def load_models():
#     import torch
#     import os

#     if not torch.cuda.is_available():
#         lp = torch.load("models/lp_pred_model.ckpt", map_location=torch.device('cpu'))
#     else:
#         lp = torch.load("models/lp_pred_model.ckpt")

@app.function(image=image)
def predict(smi_path):
    import numpy as np
    from tqdm import tqdm
    import os
    import sys

    sys.path.insert(0, "/")
    os.chdir("/")

    os.chdir("/simg")
    
    import simg.model_utils
    from simg.data import get_connectivity_info
    from simg.model_utils import pipeline

    simg.model_utils.LP_CHECKPOINT_PATH = "/models/lp_pred_model.ckpt"

    all_smiles_splitted = [l.strip() for l in open(smi_path, 'r').readlines()]
    all_smiles_splitted = [l for l in all_smiles_splitted if l]
    
    xyzs = []
    for smi in tqdm(all_smiles_splitted):
        try:
            result = smi_to_xyz(smi)
            if result:
                xyzs.append(result)
        except Exception as e:
            print(f"Error processing SMILES {smi}: {e}")
            continue

    if not xyzs:
        raise RuntimeError("No valid molecules could be processed")

    smi, mol = xyzs[0]
    print("Generated XYZ:")
    print(mol)
    
    xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
    symbols = [l.split()[0] for l in xyz_data]
    coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
    connectivity = get_connectivity_info(xyz_data)
    
    for i in range(0, len(xyzs), 10_000):
        mols = xyzs[i: i + 10_000]
        output = []
        for smi, mol in tqdm(mols):
            try:
                xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
                symbols = [l.split()[0] for l in xyz_data]
                coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
                connectivity = get_connectivity_info(xyz_data)

                output.append([
                smi, mol, pipeline(symbols, coordinates, connectivity)
                ])
            except:
                continue
    return output

#@app.function(image=image)
def smi_to_xyz(smi):
    import uuid
    import os
    import subprocess
    
    path = uuid.uuid4().hex
    xyz_path = f"{path}.xyz"
    smi_path = f"{path}.smi"

    try:
        with open(smi_path, "w") as f:
            f.write(smi + '\n')

        # Use subprocess for better error handling
        result = subprocess.run([
            'obabel', '-i', 'smi', smi_path, 
            '-o', 'xyz', '-O', xyz_path, 
            '--gen3d'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Open Babel SMILES conversion error: {result.stderr}")
            return None
        
        if not os.path.exists(xyz_path):
            print(f"XYZ file not created for SMILES: {smi}")
            return None

        with open(xyz_path, "r") as f:
            xyz = f.read()
        
        if not xyz.strip():
            print(f"Empty XYZ file for SMILES: {smi}")
            return None

        return (smi, xyz)
        
    finally:
        for temp_file in [xyz_path, smi_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.local_entrypoint()
def main():
    predict.remote("assets/mol.smi")
