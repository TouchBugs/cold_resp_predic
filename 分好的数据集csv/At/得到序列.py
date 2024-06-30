import csv
from Bio import SeqIO

# 步骤1: 读取基因列表
def read_genes(filename, label):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return {row[0]: label for row in reader}

cold_responsive_genes = read_genes('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/At/cold_responsive_genes.csv', 1)
nonresponsive_genes = read_genes('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/At/nonresponsive_genes.csv', 0)

# 合并两个字典
genes = {**cold_responsive_genes, **nonresponsive_genes}

# 步骤2: 解析GFF文件，建立gene_id到位点信息的映射
def parse_gff(gff_filename):
    gene_to_locus = {}
    with open(gff_filename) as gff_file:
        for line in gff_file:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if parts[2] == 'gene':
                gene_id = parts[-1].split(';')[0].split('=')[1]
                gene_to_locus[gene_id] = (parts[0], int(parts[3]), int(parts[4]))
    return gene_to_locus

gene_to_locus = parse_gff('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/At/Arabidopsis_thaliana.TAIR10.59.gff3')

# 步骤3: 读取FA文件并提取序列
def extract_sequence(fa_filename, gene_to_locus):
    sequences = {}
    for record in SeqIO.parse(fa_filename, "fasta"):
        for gene_id, (chrom, start, end) in gene_to_locus.items():
            if record.id == chrom:
                sequences[gene_id] = str(record.seq[start-1:end])
    return sequences

sequences = extract_sequence('genome.fa', gene_to_locus)

# 步骤4: 写入新的CSV文件
with open('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/At/data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['gene_id', 'sequence', 'label'])
    for gene_id, label in genes.items():
        if gene_id in sequences:
            writer.writerow([gene_id, sequences[gene_id], label])