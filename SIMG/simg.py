import modal
from huggingface_hub import hf_hub_download

def import_models():
    hf_hub_download(repo_id="gomesgroup/simg", filename="lp_pred_model.ckpt")
    hf_hub_download(repo_id="gomesgroup/simg", filename="nbo_pred_model.ckpt")

image = modal.Image.debian_slim().apt_install([
    "openbabel"
]).run_function(
    import_models
).pip_install([
    "torch",
    "torch_geometric",
    "numpy",
    "tqdm"
])

app = modal.App("simg")

# def get_initial_graph(symbols, coordinates, connectivity):
#     import torch
#     # one hot encoding of atoms
#     x = torch.tensor([atom_type_one_hot(atom) for atom in symbols], dtype=torch.float32)

#     # bond features
#     edge_attr = torch.tensor(
#         get_bond_features(coordinates, connectivity), dtype=torch.float32
#     )

#     # remove last column (distances do not matter for LP inference)
#     edge_attr = edge_attr[:, :-1]

#     # turn into bidirectional graph
#     connectivity = connectivity + [(j, i, k) for i, j, k in connectivity]

#     # Create graph as coordination list
#     edge_index = torch.tensor([[x[0], x[1]] for x in connectivity]).t()

#     graph = Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         symbols=symbols,  # this is not used for LP inference
#         xyz_data=torch.tensor(
#             np.array(coordinates)
#         ),  # this is not used for LP inference
#         num_nodes=len(symbols),
#     )

#     return graph


# def get_connectivity_info(xyz_data):
#     import uuid
#     import os
#     import subprocess

#     connectivity = []

#     unique_name = uuid.uuid4().hex
#     xyz_temp_filename = unique_name + ".xyz"
#     sdf_temp_filename = unique_name + ".sdf"

#     try:
#         with open(xyz_temp_filename, "w") as f:
#             f.write(f"{len(xyz_data)}\n")
#             f.write("Generated molecule\n")
#             for line in xyz_data:
#                 # Remove trailing newline if present, then add it back
#                 f.write(line.strip() + "\n")
        
#         result = subprocess.run(
#             ["obabel", xyz_temp_filename, "-O", sdf_temp_filename],
#             capture_output=True,
#             text=True
#         )
        
#         if result.returncode != 0:
#             print(f"Open Babel error: {result.stderr}")
#             raise RuntimeError(f"Open Babel failed to convert XYZ to SDF: {result.stderr}")
        
#         # Check if SDF file was created and has content
#         if not os.path.exists(sdf_temp_filename):
#             raise RuntimeError("SDF file was not created by Open Babel")
        
#         with open(sdf_temp_filename, "r") as f:
#             sdf_content = f.read()
        
#         if not sdf_content.strip():
#             raise RuntimeError("SDF file is empty")
        
#         sdf_lines = sdf_content.split('\n')
        
#         sdf_lines = [line[:3] + ' ' + line[3:] for line in sdf_lines]
#         sdf_lines = [x.strip() for x in sdf_lines][3:]
#         sdf_header = sdf_lines[0].split()
#         sdf_header = list(filter(lambda x: x, sdf_header))
#         num_atoms, num_bonds = int(sdf_header[0]), int(sdf_header[1])
#         raw_connectivity = sdf_lines[num_atoms + 1: -2]
        
#         for connection in raw_connectivity:
#             A = connection.split(" ")
#             A = [x for x in A if x != ""]
#             if A[0] == 'M':
#                 continue
#             # (source_node, target_node, bond_type)
#             connectivity.append((int(A[0]) - 1, int(A[1]) - 1, int(A[2])))
        
#         connectivity.sort(key=lambda x: (x[0], x[1]))

#     finally:
#         # Clean up temporary files
#         for temp_file in [xyz_temp_filename, sdf_temp_filename]:
#             if os.path.exists(temp_file):
#                 os.remove(temp_file)

#     return connectivity

# #@app.function(image=image)
# def predict(smi_path):
#     import numpy as np
#     from tqdm import tqdm

#     def pipeline(symbols, coordinates, connectivity):
#         molecular_graph = get_initial_graph(symbols, coordinates, connectivity)
#         n_lps, n_conj_lps = predict_lps(molecular_graph)
#         graph = get_final_graph(molecular_graph, connectivity, n_lps, n_conj_lps)
#         graph = prepare_graph(graph)
#         (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds) = make_preds_no_gt(graph, threshold, use_threshold)

#         return graph, (n_lps, n_conj_lps), (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds)

#     all_smiles_splitted = [l.strip() for l in open(smi_path, 'r').readlines()]
#     all_smiles_splitted = [l for l in all_smiles_splitted if l]
    
#     xyzs = []
#     for smi in tqdm(all_smiles_splitted):
#         try:
#             result = smi_to_xyz(smi)
#             if result:
#                 xyzs.append(result)
#         except Exception as e:
#             print(f"Error processing SMILES {smi}: {e}")
#             continue

#     if not xyzs:
#         raise RuntimeError("No valid molecules could be processed")

#     smi, mol = xyzs[0]
#     print("Generated XYZ:")
#     print(mol)
    
#     xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
#     symbols = [l.split()[0] for l in xyz_data]
#     coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
#     connectivity = get_connectivity_info(xyz_data)
    
#     for i in range(0, len(xyzs), 10_000):
#         mols = xyzs[i: i + 10_000]
#         output = []
#         for smi, mol in tqdm(mols):
#             try:
#                 xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
#                 symbols = [l.split()[0] for l in xyz_data]
#                 coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
#                 connectivity = get_connectivity_info(xyz_data)

#                 #TODO: PIPELINE
#                 print("here")
#             except:
#                 print("excepted")
#                 continue

# #@app.function(image=image)
# def smi_to_xyz(smi):
#     import uuid
#     import os
#     import subprocess
    
#     path = uuid.uuid4().hex
#     xyz_path = f"{path}.xyz"
#     smi_path = f"{path}.smi"

#     try:
#         with open(smi_path, "w") as f:
#             f.write(smi + '\n')

#         # Use subprocess for better error handling
#         result = subprocess.run([
#             'obabel', '-i', 'smi', smi_path, 
#             '-o', 'xyz', '-O', xyz_path, 
#             '--gen3d'
#         ], capture_output=True, text=True)
        
#         if result.returncode != 0:
#             print(f"Open Babel SMILES conversion error: {result.stderr}")
#             return None
        
#         if not os.path.exists(xyz_path):
#             print(f"XYZ file not created for SMILES: {smi}")
#             return None

#         with open(xyz_path, "r") as f:
#             xyz = f.read()
        
#         if not xyz.strip():
#             print(f"Empty XYZ file for SMILES: {smi}")
#             return None

#         return (smi, xyz)
        
#     finally:
#         for temp_file in [xyz_path, smi_path]:
#             if os.path.exists(temp_file):
#                 os.remove(temp_file)

# if __name__ == "__main__":
#     xyz = "assets/mol.smi"
#     predict(xyz)