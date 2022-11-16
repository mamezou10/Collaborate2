
import pandas as pd
import glob

## humanのidを準備
gtf = pd.read_csv("/mnt/Daisy/human/gencode.v42.basic.annotation.gtf", skiprows=5, header=None, sep="\t")
kari=gtf.loc[:,8].str.split("; ", expand=True)
kari=kari.fillna("kari")
for i in range(22):
    kari[i] = kari[i].mask(~kari[i].str.contains("gene_id|gene_name", regex=True))

kari = kari.fillna("")
# df = kari[0]
# for i in range(1,6):
#     df.str.cat(kari[i], sep=", ")
df = pd.DataFrame(kari[0].str.cat([kari[1], kari[2], kari[3], kari[4], kari[5], kari[6], kari[7], kari[8], kari[9], kari[10], 
                    kari[11], kari[12], kari[13], kari[14], kari[15], kari[16], kari[17], kari[18], kari[19], kari[20], kari[21]], sep=","))
df = df[0].str.replace(',| |"|gene_id', "").str.split("gene_name", expand=True)
df.columns = ["gene_id", "gene_name"]
df["gene_id"] = df["gene_id"].str.split(".", expand=True)[0]
df = df.drop_duplicates().reset_index()[["gene_id", "gene_name"]]
df.to_csv("/mnt/Daisy/human/id_name.tsv", sep="\t", header=True)

gt_ids = pd.read_csv("/mnt/Daisy/human/human_genes_transcript_id.tsv", sep="\t")
human_id = df
human_id = pd.merge(human_id, gt_ids, left_on='gene_id', right_on='Gene stable ID')
human_id["species"] = "human"

## mouseのidを準備
gt_ids = pd.read_csv("/mnt/Daisy/mouse/mouse_genes_transcripts_id.tsv", sep="\t")
mouse_id = pd.read_csv("/mnt/Daisy/mouse/id_name.tsv", sep="\t", index_col=0)
mouse_id = pd.merge(mouse_id, gt_ids, left_on='gene_id', right_on='Gene stable ID')
mouse_id["species"] = "mouse"

## mixのid表
mix_id = pd.concat([human_id, mouse_id], axis=0)

## 発現定量を拾う
files = glob.glob("/mnt/ssd8t/Projects/osawa/SAT1/*/mix/abundance.tsv")

exps = list()
for i in range(len(files)):
    kari = pd.read_csv(files[i], sep="\t")
    exp = pd.merge(kari, mix_id, left_on='target_id', right_on='Transcript stable ID version', how="outer")[['gene_name', 'species', 'est_counts']].groupby(['gene_name', 'species']).sum().reset_index()
    s_name = os.path.basename(os.path.dirname(os.path.dirname(files[i])))
    exp.columns = ["gene_name", "species", s_name]
    exps.append(exp)


exp_total = exps[0]
for i in range(1,len(files)):
    print(i)
    exp_total = pd.merge(exp_total, exps[i], on=['gene_name', 'species'], how="outer")

## どれくらい違うか　-> human    1.143782e+07, mouse    4.045590e+04 ->0.3%くらい
exp_total[["species",  "shSAT1_2_1"]].groupby("species").sum()
exp_total[["species",  "shSAT1_2_2"]].groupby("species").sum()

exp_total.loc[exp_total.species=="mouse",:]


exp_total[(exp_total.iloc[:,2:].sum(axis=1)>0)&(exp_total.species=="mouse")]





