import heapq
import os
from ultrafast import featurizers
from rdkit import Chem
from torch.nn.init import xavier_normal_

def get_featurizer(featurizer_string, *args, **kwargs):
    return getattr(featurizers, featurizer_string)(*args, **kwargs)

def xavier_normal(model):
    for name, param in model.named_parameters():
        if name.endswith('bias'):
            param.data.fill_(0)
        else:
            xavier_normal_(param.data)

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

# create a class that keeps track of the topk cosine_similarities and their IDs
class TopK:
    def __init__(self, k):
        self.K = k
        self.data = []
    def push(self, similarity, id):
        if len(self.data) < self.K:
            heapq.heappush(self.data, (similarity, id))
        elif similarity > self.data[0][0]:
            heapq.heappushpop(self.data,(similarity, id))
    def get(self):
        return sorted(self.data, reverse=True)
    # write a method that pushes a list of similarities and IDs
    def push_list(self, similarities, ids):
        for similarity, id in zip(similarities, ids):
            self.push(similarity, id.item())

def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    # check if the pdb file is empty
    assert os.path.getsize(path) > 0, f"PDB file is empty: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

# The below is copy pasted from github rdkit machine learning library # 

r"""
$Id$

Scoring - Calculate rank statistics

Created by Sereina Riniker, October 2012
after a file from Peter Gedeck, Greg Landrum

\param scores: ordered list with descending similarity containing
               active/inactive information
\param col: column index in scores where active/inactive information is stored
\param fractions: list of fractions at which the value shall be calculated
\param alpha: exponential weight
"""

import math
from collections import namedtuple


def CalcROC(scores, col):
  """ Determines a ROC curve """
  numMol = len(scores)
  if numMol == 0:
    raise ValueError('score list is empty')
  TPR = [0] * numMol  # True positive rate: TP/(TP+FN)
  FPR = [0] * numMol  # False positive rate: FP/(TN+FP)
  numActives = 0
  numInactives = 0

  # loop over score list
  for i in range(numMol):
    if scores[i][col]:
      numActives += 1
    else:
      numInactives += 1
    TPR[i] = numActives  # TP
    FPR[i] = numInactives  # FP

  # normalize, check that there are actives and inactives
  if numActives > 0:
    TPR = [1.0 * i / numActives for i in TPR]
  if numInactives > 0:
    FPR = [1.0 * i / numInactives for i in FPR]

  RocCurve = namedtuple('RocCurve', ['FPR', 'TPR'])
  return RocCurve(FPR=FPR, TPR=TPR)


def CalcAUC(scores, col):
  """ Determines the area under the ROC curve """
  # determine the ROC curve
  roc = CalcROC(scores, col)
  FPR = roc.FPR
  TPR = roc.TPR

  numMol = len(scores)
  AUC = 0

  # loop over score list
  for i in range(0, numMol - 1):
    AUC += (FPR[i + 1] - FPR[i]) * (TPR[i + 1] + TPR[i])

  return 0.5 * AUC


def _RIEHelper(scores, col, alpha):
  numMol = len(scores)
  alpha = float(alpha)
  if numMol == 0:
    raise ValueError('score list is empty')
  if alpha <= 0.0:
    raise ValueError('alpha must be greater than zero')

  denom = 1.0 / numMol * ((1 - math.exp(-alpha)) / (math.exp(alpha / numMol) - 1))
  numActives = 0
  sum_exp = 0

  # loop over score list
  for i in range(numMol):
    active = scores[i][col]
    if active:
      numActives += 1
      sum_exp += math.exp(-(alpha * (i + 1)) / numMol)

  if numActives > 0:  # check that there are actives
    RIE = sum_exp / (numActives * denom)
  else:
    RIE = 0.0

  return RIE, numActives


def CalcRIE(scores, col, alpha):
  """ RIE original definded here:
    Sheridan, R.P., Singh, S.B., Fluder, E.M. & Kearsley, S.K.
    Protocols for Bridging the Peptide to Nonpeptide Gap in Topological Similarity Searches.
    J. Chem. Inf. Comp. Sci. 41, 1395-1406 (2001).
    """
  RIE, _ = _RIEHelper(scores, col, alpha)
  return RIE


def CalcBEDROC(scores, col, alpha):
  """ BEDROC original defined here:
    Truchon, J. & Bayly, C.I.
    Evaluating Virtual Screening Methods: Good and Bad Metric for the "Early Recognition"
    Problem. J. Chem. Inf. Model. 47, 488-508 (2007).
    ** Arguments**

      - scores: 2d list or numpy array
             0th index representing sample
             scores must be in sorted order with low indexes "better"
             scores[sample_id] = vector of sample data
      -  col: int
             Index of sample data which reflects true label of a sample
             scores[sample_id][col] = True iff that sample is active
      -  alpha: float
             hyper parameter from the initial paper for how much to enrich the top
     **Returns**
       float BedROC score
    """
  # calculate RIE
  RIE, numActives = _RIEHelper(scores, col, alpha)

  if numActives > 0:
    numMol = len(scores)
    ratio = 1.0 * numActives / numMol
    RIEmax = (1 - math.exp(-alpha * ratio)) / (ratio * (1 - math.exp(-alpha)))
    RIEmin = (1 - math.exp(alpha * ratio)) / (ratio * (1 - math.exp(alpha)))

    if RIEmax != RIEmin:
      BEDROC = (RIE - RIEmin) / (RIEmax - RIEmin)
    else:  # numActives = numMol
      BEDROC = 1.0
  else:
    BEDROC = 0.0

  return BEDROC


def CalcEnrichment(scores, col, fractions):
  """ Determines the enrichment factor for a set of fractions """
  numMol = len(scores)
  if numMol == 0:
    raise ValueError('score list is empty')
  if len(fractions) == 0:
    raise ValueError('fraction list is empty')
  for i in fractions:
    if i > 1 or i < 0:
      raise ValueError('fractions must be between [0,1]')

  numPerFrac = [math.ceil(numMol * f) for f in fractions]
  numPerFrac.append(numMol)
  numActives = 0
  enrich = []

  # loop over score list
  for i in range(numMol):
    if i > (numPerFrac[0] - 1) and i > 0:
      enrich.append(1.0 * numActives * numMol / i)
      numPerFrac.pop(0)
    active = scores[i][col]
    if active:
      numActives += 1

  if numActives > 0:  # check that there are actives
    enrich = [e / numActives for e in enrich]
  else:
    enrich = [0.0] * len(fractions)
  return enrich


#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# The aboves ends the copy paste. #

