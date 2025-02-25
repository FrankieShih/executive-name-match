# executive-name-match
Economics and business scholars often struggle with matching full executive names, particularly when dealing with unstructured or inconsistent formats (e.g., abbreviations and nicknames) in executive-related research. This repository offers a easy-to-use RAG-based solution to accurately link unstructured executive names to a structured database, minimizing the manual effort required for name matching and entity resolution.

# Requirement
A structured executive dataset: a CSV file containing year, CUSIP (or GVKEY as applicable), exec_fullname, execid, and title.

## Example Usage

To find the top matching executives, use the following function:

```python
find_top_matching_execs(2004, '00790310', 'Hector Ruiz', embeddings_df, name_list)

[
    ('Hector de Jesus Ruiz, Ph.D.', 0.9239),
    ('Derrick R. Meyer', 0.6810),
    ('Iain M. Morris', 0.6336),
    ('Henri P. Richard', 0.6253),
    ('Thomas M. McCoy', 0.6251)
]
