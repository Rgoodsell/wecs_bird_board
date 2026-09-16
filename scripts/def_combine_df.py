#----- Libraries
import pandas as pd
import glob
import os
import def_combine_df as cb
#-----

# ----- function to combine .csv from named folders
def combine_df(fileout):
    path =  "/Users/robgoodsell/Library/CloudStorage/OneDrive-UWEBristol/WECS_bird_board/"
    files = glob.glob(path + "**/*.csv" , recursive=True)
    names = glob.glob(path+"*/" , recursive=True)
    names = os.listdir(path)[1:]

    flist = []
    for idx, file in enumerate(files):
        df = pd.read_csv(file, index_col=False)
        df = df.assign(Name = names[idx])  
        flist.append(df)

    print("combining files")
    df_out = pd.concat(flist, axis=0, ignore_index=False)
    df_out.to_csv(fileout)


