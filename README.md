# Exo-GenomicSavant
Attaining savant-level mastery in the manipulation of extraterrestrial genomes with AI.

# Guide 

```python
from Bio import Entrez

def retrieve_genome_sequence(genome_id):
    Entrez.email = 'your_email@example.com'  # Set your email address here
    handle = Entrez.efetch(db='nucleotide', id=genome_id, rettype='fasta', retmode='text')
    record = handle.read()
    handle.close()
    return record

# Example usage
genome_id = 'NC_045512'  # Replace with the actual genome ID
sequence = retrieve_genome_sequence(genome_id)

# Output the DNA sequence in a markdown code block
print("```")
print(sequence)
print("```")
```

Make sure to replace `'your_email@example.com'` with your actual email address. This is required by the NCBI Entrez system to identify the user and prevent abuse.
