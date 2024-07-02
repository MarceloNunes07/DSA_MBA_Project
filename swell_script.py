# Python script for data manipulation and preparation of Figures

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Load data
manobras = pd.read_excel(r"C:\OneDrive - Ceará Marine Pilots\Marcelo\MBA Data Science\TCC\manobras_swell.xlsx")
manobras_extra = pd.read_excel(r"C:\OneDrive - Ceará Marine Pilots\Marcelo\MBA Data Science\TCC\swell_2_extra_registers.xlsx")
manobras_extra['Cais'] = manobras_extra['Cais'].astype(str)
manobras = pd.concat([manobras, manobras_extra], ignore_index=True)

# Preliminary step to fill 'Cais' field for aborted maneuvers
manobras.loc[(manobras['Navio'] == 'CMA CGM CAYENNE') & (manobras['Calado'] == 6.2) & 
             (manobras['Prático'] == 'Bessa') & (manobras['Tipo Man.'] == 'M'), 'Cais'] = '105'
manobras.loc[(manobras['Navio'] == 'CHIPOLBROK GALAXY') & (manobras['Calado'] == 10.1) & 
             (manobras['Prático'] == 'Custódio') & (manobras['Tipo Man.'] == 'M'), 'Cais'] = '106'
manobras.loc[(manobras['Navio'] == 'CMM CONTINUITY') & (manobras['Calado'] == 2.6) & 
             (manobras['Prático'] == 'Pedro') & (manobras['Tipo Man.'] == 'M'), 'Cais'] = '102'

# Exclude rows where 'Swell Observado' is empty
manobras = manobras.dropna(subset=['Swell Observado'])

# Data manipulation
manobras = manobras.rename(columns={'Navio': 'ship', 'Tipo': 'ship_type', 'Calado': 'draft',
                                    'Loa': 'loa', 'Boca': 'beam', 'Dwt': 'dwt', 'Tipo Man.': 'man_type', 
                                    'Cais': 'berth', 'Maré Amplitude': 'tide_amplitude', 'Maré': 'tide', 
                                    'Noturna?': 'night', 'Vento Máximo': 'wind_max', 'Swell Observado': 'swell_obs', 
                                    'Prático': 'pilot', 'Data/Hora': 'date_time', 'Maré Altura': 'tide_height'})

manobras = manobras.drop(columns=['Porto'])

manobras['man_type'] = manobras['man_type'].replace({'E': 0, 'M': 0, 'S': 1})
manobras['night'] = manobras['night'].replace({'Não': 0, 'Sim': 1})
manobras['berth'] = manobras['berth'].replace({'P.Ext': '202', 'P.Int': '201'})

# Convert negative tide_amplitude values to positive
manobras['tide_amplitude'] = manobras['tide_amplitude'].abs()

# Generate 'month' column
manobras['month'] = pd.to_datetime(manobras['date_time']).dt.month

# Load wave data
dados_ondas = pd.read_excel(r"C:\OneDrive - Ceará Marine Pilots\Marcelo\MBA Data Science\TCC\dados_ondas_gfs_wave.xlsx")

# Simplify column names and drop 'Estação'
dados_ondas = dados_ondas.rename(columns={'Estação': 'estacao', 'Data/Hora': 'date_time', 'Altura (m)': 'wave_h', 
                                          'Altura Swell (m)': 'swell_h', 'Altura Swell 2 (m)': 'swell2_h', 
                                          'Período (s)': 'wave_p', 'Período Swell (s)': 'swell_p', 
                                          'Período Swell 2 (s)': 'swell2_p', 'Direção': 'wave_dir', 
                                          'Direção Swell': 'swell_dir', 'Direção Swell 2': 'swell2_dir', 
                                          'Setor': 'sector'})

dados_ondas = dados_ondas.drop(columns=['estacao'])
dados_ondas = dados_ondas.drop_duplicates(subset=['date_time'])

# Fill missing time intervals
interval = '30T'
dados_ondas['date_time'] = pd.to_datetime(dados_ondas['date_time'])
dados_ondas = dados_ondas.set_index('date_time').resample(interval).ffill().reset_index()

# Join dataframes
manobras_ondas = pd.merge(manobras, dados_ondas, on='date_time', how='inner')

# Circular-linear transformation for angular data
manobras_ondas['wave_dir_sin'] = np.sin(np.deg2rad(manobras_ondas['wave_dir']))
manobras_ondas['wave_dir_cos'] = np.cos(np.deg2rad(manobras_ondas['wave_dir']))
manobras_ondas['swell_dir_sin'] = np.sin(np.deg2rad(manobras_ondas['swell_dir']))
manobras_ondas['swell_dir_cos'] = np.cos(np.deg2rad(manobras_ondas['swell_dir']))
manobras_ondas['swell2_dir_sin'] = np.sin(np.deg2rad(manobras_ondas['swell2_dir']))
manobras_ondas['swell2_dir_cos'] = np.cos(np.deg2rad(manobras_ondas['swell2_dir']))

# Function to manipulate tide data
def extract_tide_info(feature):
    try:
        time1 = pd.to_datetime(feature[3:8], format='%H:%M').time()
    except:
        time1 = pd.to_datetime(feature[4:9], format='%H:%M').time()
    try:
        time2 = pd.to_datetime(feature[28:33], format='%H:%M').time()
    except:
        time2 = pd.to_datetime(feature[29:34], format='%H:%M').time()
    try:
        time3 = pd.to_datetime(feature[62:67], format='%H:%M').time()
    except:
        time3 = pd.to_datetime(feature[63:68], format='%H:%M').time()
    
    if pd.isna(time1) or pd.isna(time2) or pd.isna(time3):
        return np.nan
    
    time1 = time1.hour + time1.minute / 60
    time2 = time2.hour + time2.minute / 60
    time3 = time3.hour + time3.minute / 60
    
    if "BM" in feature[17:19]:
        status_tide = "enchente"
    else:
        status_tide = "vazante"
    
    if time2 < time1:
        dif1 = time2 + 24 - time1
    else:
        dif1 = time2 - time1
    
    if time3 < time2:
        dif2 = time3 + 24 - time2
    else:
        dif2 = time3 - time2
    
    if status_tide == "enchente" and dif1 < 1.5:
        tide = "BM"
    elif status_tide == "enchente" and dif2 < 1.5:
        tide = "PM"
    elif status_tide == "enchente":
        tide = "enchente"
    elif status_tide == "vazante" and dif1 < 1.5:
        tide = "PM"
    elif status_tide == "vazante" and dif2 < 1.5:
        tide = "BM"
    else:
        tide = "vazante"
    
    return tide

# Apply function to create 'tide_phase' feature
manobras_ondas['tide_phase'] = manobras_ondas['tide'].apply(extract_tide_info)

# Convert some features to categorical
categorical_features = ['berth', 'man_type', 'night', 'swell_obs', 'month', 'tide_phase', 'sector', 'pilot', 'ship_type']
for feature in categorical_features:
    manobras_ondas[feature] = manobras_ondas[feature].astype('category')

# Filter datasets based on swell observation
manobras_zero = manobras_ondas[manobras_ondas['swell_obs'] == 0]
manobras_01 = manobras_ondas[manobras_ondas['swell_obs'] == 1]
manobras_02 = manobras_ondas[manobras_ondas['swell_obs'] == 2]

#statistics:
manobras_zero_stats = manobras_zero.describe()
manobras_01_stats = manobras_01.describe()
manobras_02_stats = manobras_02.describe()


#%% Heuristic approach

#adding new columns to receive the classfication:
manobras_ondas['score']=None
manobras_ondas['heuristic']=None
#applying the heuristic approach to inspect the accuracy (reference to compare with the ML models performance)
for index, row in manobras_ondas.iterrows():
    direction_score=0
    period_score=0
    height_score=0
    
    if row['sector'] == 'N' or row['sector'] == 'NNW' or row['sector'] == 'NW' or row['sector'] == 'WNW':
        direction_score=2
    elif row['sector'] == 'NNE':
        direction_score=1
    else:
        direction_score=0
    
    if row['wave_p'] < 8:
        period_score=0
    elif row['wave_p'] < 10:
        period_score=1
    elif row['wave_p'] >= 10:
        period_score=2
    
    if row['wave_h'] < 1:
        height_score=0
    elif row['wave_h'] <= 1.5:
        height_score=1
    elif row['wave_h'] <= 2:
        height_score=2
    elif row['wave_h'] > 2:
        height_score=3
        
    manobras_ondas['score'][index]=direction_score+period_score+height_score
    if manobras_ondas['score'][index] <= 3:
        manobras_ondas['heuristic'][index]=0.0
    elif manobras_ondas['score'][index] <= 5:
        manobras_ondas['heuristic'][index]=1.0
    elif manobras_ondas['score'][index] > 5:
        manobras_ondas['heuristic'][index]=2.0

# Extract the columns
y_true = manobras_ondas['swell_obs']
y_pred = manobras_ondas['heuristic']

# Convert y_pred to categorical type
y_pred = y_pred.astype('category')

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Initialize dictionaries to store sensitivity and specificity
sensitivity = {}
specificity = {}

# Calculate sensitivity and specificity for each class
for i in range(conf_matrix.shape[0]):
    TP = conf_matrix[i, i]
    FN = conf_matrix[i, :].sum() - TP
    FP = conf_matrix[:, i].sum() - TP
    TN = conf_matrix.sum() - (TP + FN + FP)

    sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0

# Print the results
print("Confusion Matrix:")
print(conf_matrix)

print("\nSensitivity for each class:")
for wave_class, sens in sensitivity.items():
    print(f"Class {wave_class}: {sens}")

print("\nSpecificity for each class:")
for wave_class, spec in specificity.items():
    print(f"Class {wave_class}: {spec}")

# Compute the overall accuracy
accuracy = accuracy_score(y_true, y_pred)

#%% plots
# histogramas for each class (by month, by berth, etc.)

# try scatter plots:

custom_palette = {0.0: 'blue', 1.0: 'green',2.0:'red'}
custom_markers = {0.0: 'o', 1.0: 's',2.0:'X'}    

plt.figure(figsize=(10,10))
sns.scatterplot(data=manobras_ondas, x="wave_p", y="wave_dir_sin", 
                hue="swell_obs",style="swell_obs",s=50,
                palette=custom_palette, markers=custom_markers)
plt.title("Class distribution by wave period and wave direction sine")
plt.xlabel('Wave period',fontsize=12)
plt.ylabel('Wave direction sine',fontsize=12)
plt.legend(bbox_to_anchor=(1,1), fontsize = '10')
plt.show()

# plt.figure(figsize=(10,10))
# sns.scatterplot(data=manobras_ondas, x="wave_p", y="wave_dir_sin", 
#                 hue="swell_obs",style="swell_obs",size="swell_h",
#                 sizes=(20,200),palette=custom_palette, 
#                 markers=custom_markers,legend="full")
# plt.title("Class distribution by wave period and wave direction sine")
# plt.xlabel('Wave period',fontsize=12)
# plt.ylabel('Wave direction sine',fontsize=12)
# plt.legend(bbox_to_anchor=(1,1), fontsize = '10')
# plt.show()

plt.figure(figsize=(15,10))
sns.scatterplot(data=manobras_ondas, x="wave_p", y="swell_h", 
                hue="swell_obs",style="swell_obs",s=60,
                palette=custom_palette, markers=custom_markers)
plt.title("Class distribution by wave period and swell height")
plt.xlabel('Wave period',fontsize=12)
plt.ylabel('Swell height',fontsize=12)
plt.legend(bbox_to_anchor=(1,1), fontsize = '10')
plt.show()

# plt.figure(figsize=(10,10))
# sns.scatterplot(data=manobras_ondas, y="wave_dir_cos", x="wave_dir_sin", 
#                 hue="swell_obs",style="swell_obs",
#                 palette=custom_palette, markers=custom_markers)
# plt.title("Class distribution by wave direction (sine and cosine)")
# plt.ylabel('Wave direction cosine',fontsize=12)
# plt.xlabel('Wave direction sine',fontsize=12)
# plt.legend(bbox_to_anchor=(1,1), fontsize = '10')
# plt.show()

plt.figure(figsize=(15,10))
sns.scatterplot(data=manobras_ondas, x="swell_p", y="swell_h", 
                hue="swell_obs",style="swell_obs",s=60,
                palette=custom_palette, markers=custom_markers)
plt.title("Class distribution by swell period and swell height")
plt.xlabel('Swell period',fontsize=12)
plt.ylabel('Swell height',fontsize=12)
plt.legend(bbox_to_anchor=(1,1), fontsize = '7')
plt.show()

# plt.figure(figsize=(15,10))
# sns.scatterplot(data=manobras_ondas, x="wave_p", y="swell_h", 
#                 hue="swell_obs",style="swell_obs",size='tide_amplitude',
#                 palette=custom_palette, markers=custom_markers)
# plt.title("Class distribution by swell period and swell height")
# plt.xlabel('Wave period',fontsize=12)
# plt.ylabel('Swell height',fontsize=12)
# plt.legend(bbox_to_anchor=(1,1), fontsize = '7')
# plt.show()

#split scatter plot by months?

#filtered df for pairplot:
df_pairplot=manobras_ondas[['wave_dir_sin','wave_p','swell_h','wave_dir_cos','swell_obs']]
df_pairplot=df_pairplot.rename(columns={'wave_dir_sin':'Wave dir - Sine',
                                        'wave_p':'Wave Period',
                                        'swell_obs':'Class',
                                        'swell_h':'Swell Height',
                                        'wave_dir_cos':'Wave dir - Cosine'})

#pairplot with the 4 main numeric features (based on importance information from RF):
plt.rcParams.update({'font.size': 14}) 
pair_plot=sns.pairplot(df_pairplot, hue="Class", palette=custom_palette, markers=custom_markers)
# Get the handles and labels from the pairplot
handles = pair_plot._legend_data.values()
labels = pair_plot._legend_data.keys()
# Remove the default legend
pair_plot._legend.remove()
# Add the legend above the plots
pair_plot.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4, title='Class')
# Adjust the layout to make room for the legend
pair_plot.fig.subplots_adjust(top=0.915)  
# Show the plot
plt.show()

#another pairplot, but now with the wave direction:

#filtered df for pairplot:
df_pairplot2=manobras_ondas[['wave_dir','wave_p','swell_h','swell_obs']]
df_pairplot2=df_pairplot2.rename(columns={'wave_dir':'Wave direction',
                                        'wave_p':'Wave Period',
                                        'swell_obs':'Class',
                                        'swell_h':'Swell Height'})

#pairplot with the 4 main numeric features (based on importance information from RF):
plt.rcParams.update({'font.size': 14}) 
pair_plot2=sns.pairplot(df_pairplot2, hue="Class", palette=custom_palette, markers=custom_markers)
# Get the handles and labels from the pairplot
handles = pair_plot2._legend_data.values()
labels = pair_plot2._legend_data.keys()
# Remove the default legend
pair_plot2._legend.remove()
# Add the legend above the plots
pair_plot2.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4, title='Class')
# Adjust the layout to make room for the legend
pair_plot2.fig.subplots_adjust(top=0.88)  
# Save the plot with 600 dpi
pair_plot2.savefig(r"C:\OneDrive - Ceará Marine Pilots\Marcelo\MBA Data Science\TCC\Resultados Preliminares\pair_plot2.png", dpi=600)
# Show the plot
plt.show()    


#violin plots:
#It shows the distribution of data points after grouping by one (or more) variables. 
#Unlike a box plot, each violin is drawn using a kernel density estimate of the underlying 
#distribution (at least one of the features must be numeric).
    
    
plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_p',inner='point')
plt.title('Wave Period Distribution by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Wave Period (s)',fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_dir_sin',inner='point')
plt.title('Wave Direction Sine by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Wave Direction (Sine)',fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='swell_h',inner='point')
plt.title('Swell Height by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Swell Height (m)',fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_dir_cos',inner='point')
plt.title('Wave Direction Cosine by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Wave Direction (Cosine)',fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='swell_p',inner='point')
plt.title('Swell Period by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Swell Period (s)',fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(data=manobras_ondas, x='swell_obs', y='tide_amplitude',inner='point')
plt.title('Tide Amplitude by Sea Agitation Level',fontsize=16)
plt.xlabel('Sea Agitation Class',fontsize=14)
plt.ylabel('Tide Amplitude (m)',fontsize=14)
plt.grid(True)
plt.show()

#combined plot with 6 violinplots:
# Create a figure and a 2x3 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_p', inner='point', 
               ax=axes[0, 0],palette=custom_palette)
axes[0, 0].set_xlabel('')
axes[0, 0].set_xticklabels([])
axes[0, 0].set_ylabel('Wave Period (s)', fontsize=18)
axes[0, 0].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_dir_sin', inner='point', 
               ax=axes[0, 1],palette=custom_palette)
axes[0, 1].set_xlabel('')
axes[0, 1].set_xticklabels([])
axes[0, 1].set_ylabel('Wave Direction (Sine)', fontsize=18)
axes[0, 1].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='swell_h', inner='point', 
               ax=axes[1, 0],palette=custom_palette)
axes[1, 0].set_xlabel('')
axes[1, 0].set_xticklabels([])
axes[1, 0].set_ylabel('Swell Height (m)', fontsize=18)
axes[1, 0].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='wave_dir_cos', inner='point', 
               ax=axes[1, 1],palette=custom_palette)
axes[1, 1].set_xlabel('')
axes[1, 1].set_xticklabels([])
axes[1, 1].set_ylabel('Wave Direction (Cosine)', fontsize=18)
axes[1, 1].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='swell_p', inner='point', 
               ax=axes[2, 0],palette=custom_palette)
axes[2, 0].set_xlabel('Sea Agitation Class', fontsize=18)
axes[2, 0].set_ylabel('Swell Period (s)', fontsize=18)
axes[2, 0].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
violin = sns.violinplot(data=manobras_ondas, x='swell_obs', y='tide_amplitude', inner='point', 
               ax=axes[2, 1],palette=custom_palette)
axes[2, 1].set_xlabel('Sea Agitation Class', fontsize=18)
axes[2, 1].set_ylabel('Tide Amplitude (m)', fontsize=18)
axes[2, 1].grid(True)
for patch in violin.collections:
    patch.set_alpha(0.5)
# Display the plot
plt.show()


#Circular scatter plot:

theta = np.radians(manobras_ondas['wave_dir'])

# Calculate x and y coordinates based on wave direction and wave period
radius = manobras_ondas['wave_p']  # Use wave period as radius
y = radius * np.cos(theta)
x = radius * np.sin(theta)
swell_obs=manobras_ondas['swell_obs']

# Plotting the circular scatter plot
plt.figure(figsize=(10, 10))
for r in np.arange(5, 25, 5):  # circles every 5s
    circle = plt.Circle((0, 0), radius=r, color='gray', fill=False, linestyle='--', linewidth=1)
    plt.gca().add_artist(circle)
# Plot radial lines for the circular grid
for angle in range(0, 360, 30):  # radial lines every 30º
    plt.plot([0, 20 * np.cos(np.radians(angle))], 
             [0, 20 * np.sin(np.radians(angle))], 
             color='gray', linestyle='--', linewidth=1)
# Plot each category separately
for obs in np.unique(swell_obs):
    mask = swell_obs == obs
    plt.scatter(x[mask], y[mask], label=obs, marker=custom_markers[obs], 
                color=custom_palette[obs],alpha=0.7)
plt.title('Scatter Plot: Wave Period vs. Wave Direction')
plt.grid(False)  # Disable default grid
# Set equal scaling to ensure circular aspect
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(title='Class')
# Remove xlabel, ylabel, xticks, and yticks
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.show()

#same plot, but now using swell direction and swell period:
theta2 = np.radians(manobras_ondas['swell_dir'])

# Calculate x and y coordinates based on wave direction and wave period
radius2 = manobras_ondas['swell_p']  # Use swell period as radius
y2 = radius2 * np.cos(theta2)
x2 = radius2 * np.sin(theta2)
swell_obs=manobras_ondas['swell_obs']

# Plotting the circular scatter plot
plt.figure(figsize=(10, 10))
for r in np.arange(5, 25, 5):  # Adjust range and step as needed
    circle = plt.Circle((0, 0), radius=r, color='gray', fill=False, linestyle='--', linewidth=1)
    plt.gca().add_artist(circle)
# Plot radial lines for the circular grid
for angle in range(0, 360, 30):  # Adjust angle step as needed
    plt.plot([0, 20 * np.cos(np.radians(angle))], 
             [0, 20 * np.sin(np.radians(angle))], 
             color='gray', linestyle='--', linewidth=1)
# Plot each category separately
for obs in np.unique(swell_obs):
    mask = swell_obs == obs
    plt.scatter(x2[mask], y2[mask], label=obs, marker=custom_markers[obs], 
                color=custom_palette[obs],alpha=0.7)
plt.title('Scatter Plot: Swell Period vs. Swell Direction')
plt.grid(False)  # Disable default grid
# Set equal scaling to ensure circular aspect
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(title='Class')
# Remove xlabel, ylabel, xticks, and yticks
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.show()

#same plot, but now using swell height and wave direction:
theta3 = np.radians(manobras_ondas['wave_dir'])

# Calculate x and y coordinates based on wave direction and wave period
radius3 = manobras_ondas['swell_h']  # Use swell period as radius
y3 = radius3 * np.cos(theta3)
x3 = radius3 * np.sin(theta3)
swell_obs=manobras_ondas['swell_obs']

# Plotting the circular scatter plot
plt.figure(figsize=(10, 10))
for r in np.arange(0, 2.5, 0.5):  # Adjust range and step as needed
    circle = plt.Circle((0, 0), radius=r, color='gray', fill=False, linestyle='--', linewidth=1)
    plt.gca().add_artist(circle)
# Plot radial lines for the circular grid
for angle in range(0, 360, 30):  # Adjust angle step as needed
    plt.plot([0, 2 * np.cos(np.radians(angle))], 
             [0, 2 * np.sin(np.radians(angle))], 
             color='gray', linestyle='--', linewidth=1)
# Plot each category separately
for obs in np.unique(swell_obs):
    mask = swell_obs == obs
    plt.scatter(x3[mask], y3[mask], label=obs, marker=custom_markers[obs], 
                color=custom_palette[obs],alpha=0.7)
plt.title('Scatter Plot: Swell Height vs. Wave Direction')
plt.grid(False)  # Disable default grid
# Set equal scaling to ensure circular aspect
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(title='Class')
# Remove xlabel, ylabel, xticks, and yticks
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.show()

#%% Plots for non-numeric variables (categorical features)

#month:

# Calculate overall distribution
# overall_counts_month = manobras_ondas['month'].value_counts(normalize=True).sort_index() * 100

# Calculate distribution for each class
class_zero_counts_month = manobras_zero['month'].value_counts(normalize=True).sort_index() * 100
class_one_counts_month = manobras_01['month'].value_counts(normalize=True).sort_index() * 100
class_two_counts_month = manobras_02['month'].value_counts(normalize=True).sort_index() * 100

# Combine into a single DataFrame for plotting
distribution_df_month = pd.DataFrame({
    0.0: class_zero_counts_month,
    1.0: class_one_counts_month,
    2.0: class_two_counts_month
}).fillna(0)  # Fill NaN with 0 for missing categories

# Plotting the distributions
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in distribution_df_month.columns:
    plt.plot(distribution_df_month.index, distribution_df_month[class_name], label=class_name, color=custom_palette[class_name])
    plt.fill_between(distribution_df_month.index, distribution_df_month[class_name], alpha=0.3, color=custom_palette[class_name])

# plt.title('Percentage of occurrence of Sea Agitation Level by month',fontsize=18)
plt.xlabel('Month',fontsize=16)
plt.xticks(np.arange(13), ['','Jan', 'Feb', 'Mar','Apr','May','Jun','Jul',
                           'Aug','Sep','Oct','Nov','Dec'])
plt.ylabel('Percentage',fontsize=16)
plt.ylim([0,42])
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.show()

#berth:

# Calculate overall distribution
overall_counts_berth = manobras_ondas['berth'].value_counts(normalize=True).sort_index() * 100  
# Calculate distribution for each class
class_zero_counts_berth = manobras_zero['berth'].value_counts(normalize=True).sort_index() * 100
class_one_counts_berth = manobras_01['berth'].value_counts(normalize=True).sort_index() * 100
class_two_counts_berth = manobras_02['berth'].value_counts(normalize=True).sort_index() * 100

# Combine into a single DataFrame for plotting
distribution_df_berth = pd.DataFrame({
    'Overall data': overall_counts_berth,
    0.0: class_zero_counts_berth,
    1.0: class_one_counts_berth,
    2.0: class_two_counts_berth
}).fillna(0)  # Fill NaN with 0 for missing categories

# Define custom colors and styles
colors = {
    'Overall data': 'black',
    0.0: 'blue',
    1.0: 'green',
    2.0: 'red'
}
styles = {
    'Overall data': '--',
    0.0: '-',
    1.0: '-',
    2.0: '-'
}

# Plotting the distributions
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors
for column in distribution_df_berth.columns:
    sns.lineplot(data=distribution_df_berth[column], label=column,
                 color=colors[column], linestyle=styles[column])

# plt.title('Percentage of occurrence of Sea Agitation Level by Berth',fontsize=18)
plt.xlabel('Berth',fontsize=16)
plt.xticks(np.arange(7), ['102', '103', '104','105','106','201','202'])
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.show()

#tide phase:

# Calculate overall distribution
overall_counts_tide_phase = manobras_ondas['tide_phase'].value_counts(normalize=True).sort_index() * 100  
# Calculate distribution for each class
class_zero_counts_tide_phase = manobras_zero['tide_phase'].value_counts(normalize=True).sort_index() * 100
class_one_counts_tide_phase = manobras_01['tide_phase'].value_counts(normalize=True).sort_index() * 100
class_two_counts_tide_phase = manobras_02['tide_phase'].value_counts(normalize=True).sort_index() * 100

# Combine into a single DataFrame for plotting
distribution_df_tide_phase = pd.DataFrame({
    0.0: class_zero_counts_tide_phase,
    1.0: class_one_counts_tide_phase,
    2.0: class_two_counts_tide_phase,
    'Overall data': overall_counts_tide_phase
}).fillna(0)  # Fill NaN with 0 for missing categories

# Plotting the distributions
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
sns.lineplot(data=distribution_df_tide_phase, markers=True)
plt.title('Percentage of occurrence of Sea Agitation Level by Tide Phase',fontsize=18)
plt.xlabel('Tide Phase',fontsize=16)
plt.xticks(np.arange(4), ['Low Tide', 'High Tide', 'Rising Tide','Ebb Tide'])
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.show()

# Tide Height (bins)

# Define the bins
bins_tide_height = [0, 0.5, 1, 1.5, 2, 3.5]
bin_labels_tide_height = ['0 - 0.5', '0.5 - 1', '1 - 1.5', '1.5 - 2','>2']

# Bin the 'tide_height' column
manobras_ondas['tide_height_bin'] = pd.cut(manobras_ondas['tide_height'], bins=bins_tide_height, labels=bin_labels_tide_height, right=False, include_lowest=True)

# Calculate the count for each bin and each class
count_data_tide_height = manobras_ondas.groupby(['swell_obs', 'tide_height_bin']).size().unstack(fill_value=0)

# Convert counts to percentages for each swell_obs level
count_data_percentage_tide_height = count_data_tide_height.div(count_data_tide_height.sum(axis=1), axis=0) * 100

# Calculate the overall distribution
overall_counts_tide_height = manobras_ondas['tide_height_bin'].value_counts(normalize=True).sort_index() * 100

# Plotting the distributions with filled area under the lines
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in count_data_percentage_tide_height.index:
    plt.plot(count_data_percentage_tide_height.columns, count_data_percentage_tide_height.loc[class_name], label=class_name, 
             color=custom_palette[class_name], marker=custom_markers[class_name])
    plt.fill_between(count_data_percentage_tide_height.columns, count_data_percentage_tide_height.loc[class_name], alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
plt.plot(overall_counts_tide_height.index, overall_counts_tide_height.values, label='Overall data', color='black', linestyle='--', marker='x')
plt.fill_between(overall_counts_tide_height.index, overall_counts_tide_height.values, alpha=0.3, color='black')

plt.title('Tide Height (m) Distribution by Sea Agitation Level')
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.xlabel('Tide Height (m) Bins')
plt.show()

# combined plot with tide phase and tide height:

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# First plot
colors = ['black', custom_palette[0.0], custom_palette[1.0], custom_palette[2.0]]
styles = ['--', '-', '-', '-']
for column in distribution_df_tide_phase.columns:
    if column == 'Overall data':
        sns.lineplot(data=distribution_df_tide_phase[column], ax=axes[0], label=column,
                     color='black', linestyle='--', marker='x')
    else:
        sns.lineplot(data=distribution_df_tide_phase[column], ax=axes[0], label=column,
                     color=custom_palette[float(column)], linestyle='-', marker=custom_markers[float(column)])

axes[0].set_title('Percentage of occurrence of Sea Agitation Level by Tide Phase', fontsize=20)
axes[0].set_xlabel('Tide Phase', fontsize=18)
axes[0].set_xticks(np.arange(4))
axes[0].set_xticklabels(['Low Tide', 'High Tide', 'Rising Tide','Ebb Tide'])
axes[0].set_ylabel('Percentage', fontsize=18)
axes[0].grid(True)
axes[0].legend(title='Class', fontsize=18)

# Second plot
for class_name in count_data_percentage_tide_height.index:
    axes[1].plot(count_data_percentage_tide_height.columns, count_data_percentage_tide_height.loc[class_name], 
                 label=class_name, color=custom_palette[class_name], marker=custom_markers[class_name])
    # axes[1].fill_between(count_data_percentage_tide_height.columns, count_data_percentage_tide_height.loc[class_name], 
    #                      alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
axes[1].plot(overall_counts_tide_height.index, overall_counts_tide_height.values, 
             label='Overall data', color='black', linestyle='--', marker='x')
# axes[1].fill_between(overall_counts_tide_height.index, overall_counts_tide_height.values, alpha=0.3, color='black')

axes[1].set_title('Tide Height (m) Distribution by Sea Agitation Level', fontsize=20)
axes[1].set_xlabel('Tide Height (m) Bins', fontsize=18)
axes[1].set_ylabel('Percentage', fontsize=18)
axes[1].grid(True)
axes[1].legend(title='Class', fontsize=18)

# Adjust layout
plt.tight_layout()
plt.show()



# Ships' draft (bins)

# Define the bins
bins = [1, 3, 5, 7, 9, 11.1]
bin_labels = ['1-3', '3-5', '5-7', '7-9', '9-11']

# Bin the 'draft' column
manobras_ondas['draft_bin'] = pd.cut(manobras_ondas['draft'], bins=bins, labels=bin_labels, right=False)

# Calculate the count for each bin and each class
count_data = manobras_ondas.groupby(['swell_obs', 'draft_bin']).size().unstack(fill_value=0)

# Convert counts to percentages for each swell_obs level
count_data_percentage = count_data.div(count_data.sum(axis=1), axis=0) * 100

# Calculate the overall distribution
overall_counts = manobras_ondas['draft_bin'].value_counts(normalize=True).sort_index() * 100


# Plotting the distributions with filled area under the lines
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in count_data_percentage.index:
    plt.plot(count_data_percentage.columns, count_data_percentage.loc[class_name], label=class_name, 
             color=custom_palette[class_name], marker=custom_markers[class_name])
    plt.fill_between(count_data_percentage.columns, count_data_percentage.loc[class_name], alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
plt.plot(overall_counts.index, overall_counts.values, label='Overall data', color='black', linestyle='--', marker='x')
plt.fill_between(overall_counts.index, overall_counts.values, alpha=0.3, color='black')

plt.title('Ships Draft (m) Distribution by Sea Agitation Level')
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.xlabel('Ships Draft (m) Bins')
plt.show()

# Ships' DWT (bins)

# Define the bins
bins_dwt = [0, 15000, 30000, 45000, 60000, 90000]
bin_labels_dwt = ['0-15,000', '15,000-30,000', '30,000-45,000', '45,000-60,000', '> 60,000']

# Bin the 'dwt' column
manobras_ondas['dwt_bin'] = pd.cut(manobras_ondas['dwt'], bins=bins_dwt, labels=bin_labels_dwt, right=False, include_lowest=True)

# Calculate the count for each bin and each class
count_data_dwt = manobras_ondas.groupby(['swell_obs', 'dwt_bin']).size().unstack(fill_value=0)

# Convert counts to percentages for each swell_obs level
count_data_percentage_dwt = count_data_dwt.div(count_data_dwt.sum(axis=1), axis=0) * 100

# Calculate the overall distribution
overall_counts_dwt = manobras_ondas['dwt_bin'].value_counts(normalize=True).sort_index() * 100

# Plotting the distributions with filled area under the lines
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in count_data_percentage_dwt.index:
    plt.plot(count_data_percentage_dwt.columns, count_data_percentage_dwt.loc[class_name], label=class_name, 
             color=custom_palette[class_name], marker=custom_markers[class_name])
    plt.fill_between(count_data_percentage_dwt.columns, count_data_percentage_dwt.loc[class_name], alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
plt.plot(overall_counts_dwt.index, overall_counts_dwt.values, label='Overall data', color='black', linestyle='--', marker='x')
plt.fill_between(overall_counts_dwt.index, overall_counts_dwt.values, alpha=0.3, color='black')

plt.title('Ships DWT Distribution by Sea Agitation Level')
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.xlabel('Ships DWT Bins')
plt.show()


# Ships' loa (bins)

# Define the bins
bins_loa = [0, 75, 150, 225, 350]
bin_labels_loa = ['0-75', '75-150', '150-225', '>225']

# Bin the 'loa' column
manobras_ondas['loa_bin'] = pd.cut(manobras_ondas['loa'], bins=bins_loa, labels=bin_labels_loa, right=False, include_lowest=True)

# Calculate the count for each bin and each class
count_data_loa = manobras_ondas.groupby(['swell_obs', 'loa_bin']).size().unstack(fill_value=0)

# Convert counts to percentages for each swell_obs level
count_data_percentage_loa = count_data_loa.div(count_data_loa.sum(axis=1), axis=0) * 100

# Calculate the overall distribution
overall_counts_loa = manobras_ondas['loa_bin'].value_counts(normalize=True).sort_index() * 100

# Plotting the distributions with filled area under the lines
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in count_data_percentage_loa.index:
    plt.plot(count_data_percentage_loa.columns, count_data_percentage_loa.loc[class_name], label=class_name, 
             color=custom_palette[class_name], marker=custom_markers[class_name])
    plt.fill_between(count_data_percentage_loa.columns, count_data_percentage_loa.loc[class_name], alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
plt.plot(overall_counts_loa.index, overall_counts_loa.values, label='Overall data', color='black', linestyle='--', marker='x')
plt.fill_between(overall_counts_loa.index, overall_counts_loa.values, alpha=0.3, color='black')

plt.title('Ships LOA (m) Distribution by Sea Agitation Level')
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.xlabel('Ships LOA (m) Bins')
plt.show()

# Ships' beam (bins)

# Define the bins
bins_beam = [0, 10, 20, 30, 45]
bin_labels_beam = ['0-10', '10-20', '20-30', '>30']

# Bin the 'beam' column
manobras_ondas['beam_bin'] = pd.cut(manobras_ondas['beam'], bins=bins_beam, labels=bin_labels_beam, right=False, include_lowest=True)

# Calculate the count for each bin and each class
count_data_beam = manobras_ondas.groupby(['swell_obs', 'beam_bin']).size().unstack(fill_value=0)

# Convert counts to percentages for each swell_obs level
count_data_percentage_beam = count_data_beam.div(count_data_beam.sum(axis=1), axis=0) * 100

# Calculate the overall distribution
overall_counts_beam = manobras_ondas['beam_bin'].value_counts(normalize=True).sort_index() * 100

# Plotting the distributions with filled area under the lines
plt.figure(figsize=(15, 10))

# Plot each distribution with custom colors, markers, and filled areas
for class_name in count_data_percentage_beam.index:
    plt.plot(count_data_percentage_beam.columns, count_data_percentage_beam.loc[class_name], label=class_name, 
             color=custom_palette[class_name], marker=custom_markers[class_name])
    plt.fill_between(count_data_percentage_beam.columns, count_data_percentage_beam.loc[class_name], alpha=0.3, color=custom_palette[class_name])

# Plot the overall distribution
plt.plot(overall_counts_beam.index, overall_counts_beam.values, label='Overall data', color='black', linestyle='--', marker='x')
plt.fill_between(overall_counts_beam.index, overall_counts_beam.values, alpha=0.3, color='black')

plt.title('Ships Beam (m) Distribution by Sea Agitation Level')
plt.ylabel('Percentage',fontsize=16)
plt.grid(True)
plt.legend(title='Class',fontsize=16)
plt.xlabel('Ships Beam (m) Bins')
plt.show()

# grouping the above plots regarding ships' dimensions into a single plot:

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

# Draft subplot
axes[0].set_title('Ships Draft (m) Distribution by Sea Agitation Level')
for class_name in count_data_percentage.index:
    axes[0].plot(count_data_percentage.columns, count_data_percentage.loc[class_name], 
                 label=class_name, color=custom_palette[class_name], marker=custom_markers[class_name])
    axes[0].fill_between(count_data_percentage.columns, count_data_percentage.loc[class_name], 
                         alpha=0.3, color=custom_palette[class_name])
axes[0].plot(overall_counts.index, overall_counts.values, label='Overall data', 
             color='black', linestyle='--', marker='x')
axes[0].fill_between(overall_counts.index, overall_counts.values, alpha=0.3, color='black')
axes[0].set_xlabel('Ships Draft (m) Bins',fontsize=18)
axes[0].set_ylabel('Percentage',fontsize=18)
axes[0].grid(True)

# DWT subplot
axes[1].set_title('Ships DWT Distribution by Sea Agitation Level')
for class_name in count_data_percentage_dwt.index:
    axes[1].plot(count_data_percentage_dwt.columns, count_data_percentage_dwt.loc[class_name], 
                 label=class_name, color=custom_palette[class_name], marker=custom_markers[class_name])
    axes[1].fill_between(count_data_percentage_dwt.columns, count_data_percentage_dwt.loc[class_name], 
                         alpha=0.3, color=custom_palette[class_name])
axes[1].plot(overall_counts_dwt.index, overall_counts_dwt.values, label='Overall data', 
             color='black', linestyle='--', marker='x')
axes[1].fill_between(overall_counts_dwt.index, overall_counts_dwt.values, alpha=0.3, color='black')
axes[1].set_xlabel('Ships DWT Bins',fontsize=18)
axes[1].set_ylabel('Percentage',fontsize=18)
axes[1].grid(True)

# LOA subplot
axes[2].set_title('Ships LOA (m) Distribution by Sea Agitation Level')
for class_name in count_data_percentage_loa.index:
    axes[2].plot(count_data_percentage_loa.columns, count_data_percentage_loa.loc[class_name], 
                 label=class_name, color=custom_palette[class_name], marker=custom_markers[class_name])
    axes[2].fill_between(count_data_percentage_loa.columns, count_data_percentage_loa.loc[class_name], 
                         alpha=0.3, color=custom_palette[class_name])
axes[2].plot(overall_counts_loa.index, overall_counts_loa.values, label='Overall data', 
             color='black', linestyle='--', marker='x')
axes[2].fill_between(overall_counts_loa.index, overall_counts_loa.values, alpha=0.3, color='black')
axes[2].set_xlabel('Ships LOA (m) Bins',fontsize=18)
axes[2].set_ylabel('Percentage',fontsize=18)
axes[2].grid(True)

# Beam subplot
axes[3].set_title('Ships Beam (m) Distribution by Sea Agitation Level')
for class_name in count_data_percentage_beam.index:
    axes[3].plot(count_data_percentage_beam.columns, count_data_percentage_beam.loc[class_name], 
                 label=class_name, color=custom_palette[class_name], marker=custom_markers[class_name])
    axes[3].fill_between(count_data_percentage_beam.columns, count_data_percentage_beam.loc[class_name], 
                         alpha=0.3, color=custom_palette[class_name])
axes[3].plot(overall_counts_beam.index, overall_counts_beam.values, label='Overall data', 
             color='black', linestyle='--', marker='x')
axes[3].fill_between(overall_counts_beam.index, overall_counts_beam.values, alpha=0.3, color='black')
axes[3].set_xlabel('Ships Beam (m) Bins',fontsize=18)
axes[3].set_ylabel('Percentage',fontsize=18)
axes[3].grid(True)

# Adjust legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Sea Agitation Level', loc='upper center', 
           ncol=4,fontsize=16,title_fontsize='16', handletextpad=1.5, borderaxespad=0.1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




