from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from project_metaphor import PROJECT_ROOT
import pandas as pd
import os 

DATASETS_PATH = os.path.join(PROJECT_ROOT,'datasets/')
MODELS_PATH = os.path.join(PROJECT_ROOT,'models/')
df = pd.read_csv(DATASETS_PATH+'metaphor-test.csv')

highlighted_classes=['Beneficial',
                        'Protection',
                        'Threat',
                        'Threatened',
                        'Agent',
                        'Barrier',
                        'Change',
                        'Change Type',
                        'Goal',
                        'Vehicle',
                        'Aid',
                        'Conflict',
                        'Enemy',
                        'Enemy/Side',
                        'Loser',
                        'Side',
                        'Winner',
                        'God',
                        'Legit',
                        'Worshipper',
                        'Degree',
                        'Leader',
                        'Servant',
                        'Business',
                        'Product',
                        'Built',
                        'Component',
                        'Destruction Potential',
                        'Engineer',
                        'Facilitator',
                        'Extractor',
                        'Resource',
                        'Crime',
                        'Criminal',
                        'Enforcer',
                        'Property',
                        'Punished',
                        'Punishment',
                        'Right',
                        'Victim',
                        'Experiment',
                        'Failed Experiment',
                        'Laboratory',
                        'Scientist',
                        'Topic',
                        'Beneficial Plant',
                        'Farmer/Gardener',
                        'Undesirable Plant',
                        'Domesticated Animal',
                        'Wild Animal',
                        'Child',
                        'Entity',
                        'Entity Age',
                        'Parent',
                        'Light Level',
                        'Relationship',
                        'Destination',
                        'Level',
                        'Movement',
                        'Scale',
                        'Body of Water',
                        'Flowing',
                        'Rain',
                        'Size',
                        'Fire',
                        'Torch',
                        'Weather',
                        'Weather Type',
                        'Body',
                        'Clothing',
                        'Clothing Type',
                        'Body Part',
                        'Part Type',
                        'Posture',
                        'Cold',
                        'Cooling',
                        'Hot',
                        'Warming']

model = SentenceTransformer('all-MiniLM-L6-v2')

def choose_aspect(sentence_embedding,target_embeddings):
    temp = dict()
    for k,v in target_embeddings.items():
        cos_sim = util.cos_sim(sentence_embedding, v)
        temp[k] = cos_sim
    choice = min(temp, key=temp.get)
    return choice

target_embeddings = {each:model.encode(each) for each in highlighted_classes}
sentences = df['Sentence'].to_list()
y_test = df['Schema Slot'].to_list()

y_pred = list()

for each in sentences:
    encd = model.encode(each)
    choice = choose_aspect(encd,target_embeddings)
    y_pred.append(choice)

acc = accuracy_score(y_test,y_pred)