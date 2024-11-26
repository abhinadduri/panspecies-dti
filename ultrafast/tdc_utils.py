import requests
import os
from multiprocessing import current_process
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from ultrafast.utils import get_struc_seq

def get_saprot_seq(target_id):
    process_id = current_process().pid
    request = requests.get(f"https://www.alphafold.ebi.ac.uk/api/prediction/{target_id}?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94")
    if request.status_code == 200 and 'pdbUrl' in request.json()[0]:
        pdbUrl = request.json()[0]['pdbUrl']
        pdb = requests.get(pdbUrl)
        with open(f"{target_id}.pdb", "w") as f:
            f.write(pdb.text)
        parsed_seqs = get_struc_seq('foldseek', f"{target_id}.pdb", ["A"], process_id=process_id, plddt_mask=True)
        if isinstance(parsed_seqs,dict) and 'A' in parsed_seqs:
            parsed_seqs = parsed_seqs['A']
        os.remove(f"{target_id}.pdb")
    else:
        return (target_id, None)
    return (target_id, parsed_seqs[2])

def compute_ESM_features(target_id_dict):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.esm = model.esm.half()
    model = model.to("cuda:0")
    model.trunk.set_chunk_size(64)
    esm_struct_dict = {}
    for target_id, sequence in target_id_dict.items():
        tokenized_sequence = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"]
        tokenized_sequence = tokenized_sequence.to("cuda:0")
        with torch.no_grad():
            output = model(tokenized_sequence)
        pdb = convert_outputs_to_pdb(output)
        with open(f"{target_id}.pdb", "w") as f:
            f.write(pdb)
        parsed_seqs = get_struc_seq('foldseek', f"{target_id}.pdb", ["A"], process_id=process_id, plddt_mask=True)
        if isinstance(parsed_seqs,dict) and 'A' in parsed_seqs:
            parsed_seqs = parsed_seqs['A']
        os.remove(f"{target_id}.pdb")
        esm_struct_dict[target_id] = parsed_seqs[2]
    return esm_struct_dict
    
# from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb
def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i]*100, #esm model outputs plddt scores in range [0,1]
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs
