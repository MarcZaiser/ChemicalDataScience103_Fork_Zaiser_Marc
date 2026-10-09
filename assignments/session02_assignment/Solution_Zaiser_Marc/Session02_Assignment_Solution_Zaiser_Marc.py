from typing import Dict

def parse_molecular_formula(formula: str) -> Dict[str, int]:
    atom_dict = {}
    
    i = 0 
    while i < len(formula):
        if formula[i].isupper():    
            element = formula[i]
            i += 1
            
            if formula[i] < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            
        count_str =""
        while i < len(formula) and formula[i].isdigit():
            count_str += formula[i]   
            i += 1
            
        count = int(count_str) if count_str else 1
        
        if element in atom_dict:
            atom_dict[element] += count
        else:
            atom_dict[element] = count  
            
    return atom_dict    
            
                 