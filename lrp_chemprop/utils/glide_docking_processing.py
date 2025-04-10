import os
import pandas as pd


folder_name = 'glide-dock_SP_'
job_list = input('Insert Job List:')
job_list = job_list.split(", ")

def glide_result(folder_name,job_list):
    for job in job_list:
        file_name = folder_name+str(job)+'/'+folder_name+str(job)+'.csv'
        df = pd.read_csv(file_name)
        df_filter = df[['title','r_i_docking_gscore']]
        df_filter.columns = ['id','docking_score']
        df_sort = df_filter.sort_values(['id','docking_score'], ascending=True)
        df_unique = df_sort.drop_duplicates(subset=['id'],keep='first')
        df_unique = df_unique.sort_values(['docking_score'], ascending = True)

        df_unique.to_csv(folder_name+str(job)+'_processed.csv', index=False)
        

    return df_unique


if __name__ == "__main__":
    glide_result()
