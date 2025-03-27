import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def glide_monitor(job_list, directory = 'glide-dock_SP_', file_name='glide-dock_SP_'):
    '''A function to generate data from glide.log
     
    Parameters:
    ----------
    job_list (int or str): list of running jobs.
    directory (str): directory that contain the log file.
    file(str): log file name

    Returns:
    ----------
    df (pd.DataFrame): dataframe that monitoring glide docking.
     '''
    
    df = {'job': [], 'completed': [], 'total':[]}
    for job in job_list: 
        directory_name = 'glide-dock_SP_'+str(job)
        file_name = 'glide-dock_SP_'+str(job)+'.log'
        
        with open(directory_name+'/'+file_name, 'r') as f: 
            for i, line in enumerate(f.readlines()):
                if line.startswith('Number of jobs:'):
                    total_jobs = int(line.split()[3])
        
            completed_jobs = int(line.split()[0])
            
        df['job'].append(directory_name)
        df['completed'].append(completed_jobs)
        df['total'].append(total_jobs)

    df = pd.DataFrame(df)
    df['completed_fraction'] = df['completed']/df['total']
    df['running_fraction'] = 1.0 - df['completed_fraction']

    return df

def visualize_glide_monitor(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, y='job', x='completed_fraction', color='skyblue', label='Completed')
    sns.barplot(data=df, y='job', x='running_fraction', left=df['completed_fraction'], color ='salmon',label='Running')

    plt.xlabel("Complete Status")
    plt.ylabel("Jobs")
    plt.legend()
    plt.tight_layout()
    plt.show()
