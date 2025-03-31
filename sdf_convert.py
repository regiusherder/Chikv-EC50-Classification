import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def generate_conformers(smiles_list, names_list, output_path, num_confs=10):
    """Generates and optimizes 3D conformers for given SMILES and saves the best conformers in an SDF file."""
    w = Chem.SDWriter(output_path)
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES for {names_list[i]}")
                continue
            mol = Chem.AddHs(mol)
            
            conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
            conf_energies = []
            
            for conf_id in conformer_ids:
                energy = None
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000, mmffVariant='MMFF94')
                    if result == 0:
                        properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
                        ff = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=conf_id)
                        energy = ff.CalcEnergy()
                
                if energy is None:
                    result = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
                    if result == 0:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        energy = ff.CalcEnergy()
                
                if energy is not None:
                    conf_energies.append((conf_id, energy))
            
            if conf_energies:
                best_conf_id = min(conf_energies, key=lambda x: x[1])[0]
                mol.SetProp('_Name', str(names_list[i]))
                mol.SetProp('SMILES', smiles)
                w.write(mol, confId=best_conf_id)
            else:
                print(f"Failed to optimize any conformer for {names_list[i]}")
        except Exception as e:
            print(f"Error in {names_list[i]}: {e}")
    
    w.close()

def main():
    parser = argparse.ArgumentParser(description="Generate optimized 3D conformers from SMILES and save as SDF.")
    parser.add_argument('-s', '--smiles', type=str, help="Single SMILES string to process.")
    parser.add_argument('-f', '--file', type=str, help="Path to a text file containing SMILES (one per line).")
    parser.add_argument('-o', '--output', type=str, default="output.sdf", help="Path to output SDF file.")
    
    args = parser.parse_args()
    
    if not args.smiles and not args.file:
        print("Error: Provide either a SMILES string (-s) or a file containing SMILES (-f).")
        return
    
    smiles_list, names_list = [], []
    
    if args.smiles:
        smiles_list.append(args.smiles)
        names_list.append("Molecule_1")
    elif args.file:
        if not os.path.exists(args.file):
            print("Error: Input file does not exist.")
            return
        
        with open(args.file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    smiles_list.append(line)
                    names_list.append(f"Molecule_{i+1}")
    
    generate_conformers(smiles_list, names_list, args.output)
    print(f"Process completed. Output saved to {args.output}")
    
if __name__ == "__main__":
    main()