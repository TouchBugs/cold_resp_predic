# import csv
# from Bio import SeqIO

# # 步骤1: 读取基因列表
# def read_genes(filename, label):
#     with open(filename, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         return {row[0]: label for row in reader}

# cold_responsive_genes = read_genes('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/cold_responsive_genes.csv', 1)
# nonresponsive_genes = read_genes('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/nonresponsive_genes.csv', 0)

# # 合并两个字典
# genes = {**cold_responsive_genes, **nonresponsive_genes}

# def parse_gff(gff_filename):
#     gene_to_locus = {}
#     with open(gff_filename) as gff_file:
#         for line in gff_file:
#             if line.startswith('#'):
#                 continue
#             parts = line.strip().split('\t')
#             if parts[2] == 'gene':
#                 attributes = parts[-1]
#                 gene_id = None
#                 for attribute in attributes.split(';'):
#                     if attribute.startswith('ID='):
#                         gene_id = attribute.split('=')[1]
#                         break
#                 if gene_id:
#                     gene_to_locus[gene_id] = (parts[0], int(parts[3]), int(parts[4]))
#     return gene_to_locus

# gene_to_locus = parse_gff('/Data4/gly_wkdir/coldgenepredict/raw_sec/Arabidopsis/reference/Zm/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3')

# # 步骤3: 读取FA文件并提取序列
# def extract_sequence(fa_filename, gene_to_locus):
#     sequences = {}
#     for record in SeqIO.parse(fa_filename, "fasta"):
#         for gene_id, (chrom, start, end) in gene_to_locus.items():
#             if record.id == chrom:
#                 sequences[gene_id] = str(record.seq[start-1:end])
#     return sequences

# sequences = extract_sequence('/Data4/gly_wkdir/coldgenepredict/raw_sec/Arabidopsis/reference/Zm/Zm-B73-REFERENCE-NAM-5.0.fa', gene_to_locus)

# # 步骤4: 写入新的CSV文件
# with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/data.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['gene_id', 'sequence', 'label'])
#     for gene_id, label in genes.items():
#         if gene_id in sequences:
#             writer.writerow([gene_id, sequences[gene_id], label])
# 步骤5：把生成的CSV文件按照sequence的长度从长到短排序

import pandas as pd
csvfile = pd.read_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/data.csv', header=0)
csvfile['length'] = csvfile['sequence'].apply(len)
# csvfile = csvfile.sort_values(by='length', ascending=False) # 降序排列
csvfile = csvfile.sort_values(by='length') # 升序排列 
csvfile = csvfile.drop(columns='length')
# 把label列全部转为整数
csvfile['label'] = csvfile['label'].astype(int)
csvfile.to_csv('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/sorted_data.csv', index=False)



