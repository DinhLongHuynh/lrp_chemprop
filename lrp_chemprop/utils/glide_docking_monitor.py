import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def glide_docking_monitor(job_list, job_name = 'glide-dock_SP'):
    '''A function to generate data from glide.log
     
    Parameters:
    ----------
    job_list (int or list): numbner of running jobs.
    job_name (str): shared docking job name (or path). Default: glide-dock_SP

    Returns:
    ----------
    df (pd.DataFrame): dataframe that monitoring glide docking.
     '''
    
    df = {'job': [], 'completed': [], 'total':[]}
    for job in job_list: 
        directory_name = job_name + '_' +str(job)
        file_name = job_name + '_' + str(job) +'.log'
        
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


def visualize_glide_docking_monitor(df, **kwargs):
    plt.figure(figsize=kwargs.get('figsize'))
    sns.barplot(data=df, y='job', x='completed_fraction', color='skyblue', label='Completed')
    sns.barplot(data=df, y='job', x='running_fraction', left=df['completed_fraction'], color ='salmon',label='Running')

    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))
    plt.title(kwargs.get('title'))
    plt.legend()
    plt.tight_layout()
    plt.show()
